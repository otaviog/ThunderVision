#include <math_constants.h>
#include "dim.hpp"
#include "dsimem.hpp"
#include "dsimemutil.h"
#include "benchmark.hpp"
#include "cudaconstraits.hpp"

#define MAX_DISP 256

#define min4(a, b, c, d) min(a, min(b, min(c, d)))

__global__ void wtaKernel(const DSIDim dim, 
                          const float *dsi, 
                          const uint maxOffset, 
                          float *outimg);

__global__ void zeroDSIVolume(const DSIDim dsiDim,
                           float *dsiVol)
{
  const uint x = blockDim.x*blockIdx.x + threadIdx.x;
  const uint y = blockDim.y*blockIdx.y + threadIdx.y;

  if ( x < dsiDim.x && y < dsiDim.y ) {
    for (uint z=0; z<dsiDim.z; z++) {
      dsiVol[dsiOffset(dsiDim, x, y, z)] = 0.0f;
    }
  }
}

const float P1 = 30.0f/255.0f;
const float P2 = 150.0f/255.0f;

inline __device__ float pathCost(const DSIDim dsiDim, const float *costDSI,
                                 float lcD, float lcDm1, float lcDp1,
                                 uint x, uint y, uint z,
                                 float minDisp,
                                 float *aggregVol)
{
  const uint dsiOff = dsiOffset(dsiDim, x, y, z);
  const float cost = costDSI[dsiOff];

  const float Lr = cost
    + min4(lcD,
           lcDm1 + P1,
           lcDp1 + P1,
           minDisp + P2) - minDisp;

  aggregVol[dsiOff] += Lr;

  return Lr;
}

__global__ void semiglobalAggregVolKernel(const DSIDim dsiDim,
                                          const uint width,
                                          const float *costDSI,
                                          const int *matrix,
                                          const int *invMatrix,
                                          float *aggregDSI)
{
  const uint z = threadIdx.x;
  const uint y = blockIdx.x;
  const uint dz = z + 1;

  __shared__ float prevDisps[2][MAX_DISP + 2];
  __shared__ float prevDispsMin[2][MAX_DISP + 2];
  __shared__ float fMinDisp, iMinDisp;

  if ( z == 0 ) {
      prevDisps[0][0] = CUDART_INF_F;
      prevDisps[0][MAX_DISP + 1] = CUDART_INF_F;

      prevDisps[1][0] = CUDART_INF_F;
      prevDisps[1][MAX_DISP + 1] = CUDART_INF_F;
  }

  int fX, fY, iX, iY;

  fX = matrix[1]*y + matrix[2];
  fY = matrix[4]*y + matrix[5];

  iX = invMatrix[1]*y + invMatrix[2];
  iY = invMatrix[4]*y + invMatrix[5];

  uint initialOff = dsiOffset(dsiDim, fX, fY, z);
  float initialCost = costDSI[initialOff];

  prevDisps[0][dz] = initialCost;
  aggregDSI[initialOff] = initialCost;

  initialOff = dsiOffset(dsiDim, iX, iY, z);
  initialCost = costDSI[initialOff];

  prevDisps[1][dz] = initialCost;
  aggregDSI[initialOff] = initialCost;

  __syncthreads();

  float fLr, iLr;

  for (uint x=1; x<width; x++) {
#if 1
    int i = dsiDim.z >> 1;
    while ( i != 0 ) {
      if ( z < i ) {
        prevDispsMin[0][dz] = min(prevDispsMin[0][dz], prevDispsMin[0][dz + i]);
        prevDispsMin[1][dz] = min(prevDispsMin[1][dz], prevDispsMin[1][dz + i]);
      }
      __syncthreads();
      i = i >> 1;
    }

    if ( z == 0 ) {
      fMinDisp = prevDispsMin[0][1];
      iMinDisp = prevDispsMin[1][1];
    }

    __syncthreads();
#endif

    fX = matrix[0]*x + matrix[1]*y + matrix[2];
    fY = matrix[3]*x + matrix[4]*y + matrix[5];

    iX = invMatrix[0]*x + invMatrix[1]*y + invMatrix[2];
    iY = invMatrix[3]*x + invMatrix[4]*y + invMatrix[5];

    fLr = pathCost(dsiDim, costDSI,
                   prevDisps[0][dz],
                   prevDisps[0][dz - 1],
                   prevDisps[0][dz + 1],
                   fX, fY, z, fMinDisp, aggregDSI);

    iLr = pathCost(dsiDim, costDSI,
                   prevDisps[1][dz],
                   prevDisps[1][dz - 1],
                   prevDisps[1][dz + 1],
                   iX, iY, z, iMinDisp, aggregDSI);

    __syncthreads();

    prevDisps[0][dz] = fLr;
    prevDisps[1][dz] = iLr;

    __syncthreads();
  }
}

static cudaError_t copyMatrixToDev(const int matrix[2][3], int **matrix_d)
{ 
  cudaError_t r = 
    cudaMalloc(matrix_d, sizeof(int)*6);

  if ( r == cudaSuccess )
    r = cudaMemcpy(*matrix_d, matrix, sizeof(int)*6,
                   cudaMemcpyHostToDevice);
  
  return r;
}

void RunSemiGlobalDev(const tdv::Dim &tdv_dsiDim, float *dsi, float *dispImg)
{
  DSIDim dsiDim(DSIDimCreate(tdv_dsiDim));

  const int leftRightM[2][3] = {
    {1, 0, 0},
    {0, 1, 0},
  };

  const int rightLeftM[2][3] = {
    {-1, 0, dsiDim.x - 1},
    {0, 1, 0},
  };

  const int topBottomM[2][3] = {
    {0, 1, 0},
    {1, 0, 0}
  };

  const int bottomTopM[2][3] = {
    {0, -1, dsiDim.y - 1},
    {1, 0, 0}
  };

  tdv::CUerrExp cuerr;

  tdv::CudaConstraits constraits;

  float *aggregDSI;
  cuerr << cudaMalloc((void**) &aggregDSI, tdv_dsiDim.size()*sizeof(float));

  int *leftRightM_d, *rightLeftM_d, *topBottomM_d, *bottomTopM_d;
  cuerr << copyMatrixToDev(leftRightM, &leftRightM_d);  
  cuerr << copyMatrixToDev(rightLeftM, &rightLeftM_d);
  cuerr << copyMatrixToDev(topBottomM, &topBottomM_d);
  cuerr << copyMatrixToDev(bottomTopM, &bottomTopM_d);

  tdv::CudaBenchmarker bm;
  bm.begin();

  tdv::WorkSize wsz = constraits.imageWorkSize(tdv_dsiDim);

  zeroDSIVolume<<<wsz.blocks, wsz.threads>>>(dsiDim, aggregDSI);

  cuerr = cudaThreadSynchronize();

  if ( !cuerr.good() ) {
    cudaFree(aggregDSI);
    cuerr.checkErr();
  }

  semiglobalAggregVolKernel<<<dsiDim.y, dsiDim.z>>>(dsiDim, dsiDim.x,
                                                    dsi, leftRightM_d, rightLeftM_d,
                                                    aggregDSI);
  
  semiglobalAggregVolKernel<<<dsiDim.y, dsiDim.z>>>(dsiDim, dsiDim.x,
                                                    dsi, topBottomM_d, bottomTopM_d,
                                                    aggregDSI);

  cuerr = cudaThreadSynchronize();

  if ( !cuerr.good() ) {
    cudaFree(aggregDSI);
    cuerr.checkErr();
  }

#if 1

  wtaKernel<<<wsz.blocks, wsz.threads>>>(dsiDim, aggregDSI,
                                         dsiDim.x*dsiDim.y,
                                         dispImg);
#endif
  bm.end();

  cuerr = cudaFree(aggregDSI);
  cuerr.checkErr();
}
