#include <math_constants.h>
#include "dim.hpp"
#include "dsimemutil.h"
#include "cudaconstraits.hpp"

#include <iostream>

#define MAX_DISP 768

#define min4(a, b, c, d) min(a, min(b, min(c, d)))

static inline size_t diagRSizeX(int h, int x)
{
  return std::min(x + 1, h);
}

static inline size_t diagRSizeY(int w, int h, int y)
{
  return std::min(h - y, w);    
}                              

void showDiagImg(int w, int h, int *st, int *ed);
                        
__global__ void wtaKernel(const DSIDim dim,
                          const float *dsi,
                          const uint width,
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

const float P1 = 40.0f/255.0f;
const float P2 = 180.0f/255.0f;

inline __device__ float pathCost(const DSIDim dsiDim, const float *costDSI,
                                 float lcD, float lcDm1, float lcDp1,
                                 uint x, uint y, uint z,
                                 float minDisp,
                                 float P2Adjust,
                                 float *aggregVol)
{
  const uint dsiOff = dsiOffset(dsiDim, x, y, z);
  const float cost = costDSI[dsiOff];

  const float Lr = cost
    + min4(lcD,
           lcDm1 + P1,
           lcDp1 + P1,
           minDisp + P2Adjust) - minDisp;

  aggregVol[dsiOff] += Lr;

  return Lr;
}

__constant__ int matrix[6];
__constant__ int invMatrix[6];

#define MAX_ROWS 2048

__constant__ int rowsWidths[MAX_ROWS];
__constant__ int rowsStart[MAX_ROWS][2];
__constant__ int rowsEnd[MAX_ROWS][2];

__global__ void semiglobalAggregVolKernel(const DSIDim dsiDim,
                                          const float *costDSI,
                                          const float *iImage,
                                          int pathDirX, int pathDirY,
                                          float *aggregDSI)
{
  const uint z = threadIdx.x;
  const uint y = blockIdx.x;
  const uint dz = z + 1;

  __shared__ float prevDisps[2][MAX_DISP + 2];
  __shared__ float prevDispsMin[2][MAX_DISP];
  __shared__ float fMinDisp, iMinDisp;
  __shared__ float fP2Adjust, iP2Adjust,
    fLastIntesity, iLastIntesity;

  __shared__ int fX, fY, iX, iY;

  if ( z == 0 ) {
    fX = rowsStart[y][0];
    fY = rowsStart[y][1];

    iX = rowsEnd[y][0];
    iY = rowsEnd[y][1];

    prevDisps[0][0] = CUDART_INF_F;
    prevDisps[0][dsiDim.z + 1] = CUDART_INF_F;

    prevDisps[1][0] = CUDART_INF_F;
    prevDisps[1][dsiDim.z + 1] = CUDART_INF_F;

    fLastIntesity = iImage[fY*dsiDim.x + fX];
    iLastIntesity = iImage[iY*dsiDim.x + iX];
  }

  __syncthreads();

  uint initialOff = dsiOffset(dsiDim, fX, fY, z);
  float initialCost = costDSI[initialOff];

  prevDisps[0][dz] = initialCost;
  prevDispsMin[0][z] = initialCost;
  aggregDSI[initialOff] = initialCost;

  initialOff = dsiOffset(dsiDim, iX, iY, z);
  initialCost = costDSI[initialOff];

  prevDisps[1][dz] = initialCost;
  prevDispsMin[1][z] = initialCost;
  aggregDSI[initialOff] = initialCost;

  __syncthreads();

  float fLr, iLr;

  for (uint x=1; x<rowsWidths[y]; x++) {    
    int i = dsiDim.z >> 1;
    while ( i != 0 ) {
      if ( z < i ) {
        prevDispsMin[0][z] = min(prevDispsMin[0][z],
                                 prevDispsMin[0][z + i]);
        prevDispsMin[1][z] = min(prevDispsMin[1][z],
                                 prevDispsMin[1][z + i]);
      }
      __syncthreads();
      i = i >> 1;
    }

    if ( z == 0 ) {
      fMinDisp = prevDispsMin[0][0];
      iMinDisp = prevDispsMin[1][0];

      fX = fX + pathDirX;
      fY = fY + pathDirY;

      iX = iX - pathDirX;
      iY = iY - pathDirY;

      const float fI = iImage[fY*dsiDim.x + fX];
      const float iI = iImage[iY*dsiDim.x + iX];

      fP2Adjust = P2/abs(fI - fLastIntesity);
      iP2Adjust = P2/abs(iI - iLastIntesity);

      fLastIntesity = fI;
      iLastIntesity = iI;
    }

    __syncthreads();

    fLr = pathCost(dsiDim, costDSI,
                   prevDisps[0][dz],
                   prevDisps[0][dz - 1],
                   prevDisps[0][dz + 1],
                   fX, fY, z, fMinDisp, fP2Adjust,
                   aggregDSI);

    iLr = pathCost(dsiDim, costDSI,
                   prevDisps[1][dz],
                   prevDisps[1][dz - 1],
                   prevDisps[1][dz + 1],
                   iX, iY, z, iMinDisp, iP2Adjust,
                   aggregDSI);    

    __syncthreads();

    prevDisps[0][dz] = prevDispsMin[0][z] = fLr;
    prevDisps[1][dz] = prevDispsMin[1][z] = iLr;

    __syncthreads();
  }
}

void RunSemiGlobalDev(const tdv::Dim &tdv_dsiDim, const float *dsi,
                      const float *lorigin, float *aggregDSI, float *dispImg)
{
  DSIDim dsiDim(DSIDimCreate(tdv_dsiDim));
  
  tdv::CUerrExp cuerr;
  tdv::CudaConstraits constraits;
    
  tdv::WorkSize wsz = constraits.imageWorkSize(tdv_dsiDim);
  int rowsWidths_h[MAX_ROWS];
  int rowsStart_h[MAX_ROWS][2];
  int rowsEnd_h[MAX_ROWS][2];
  
  zeroDSIVolume<<<wsz.blocks, wsz.threads>>>(dsiDim, aggregDSI);      
  
  for (size_t i=0; i<dsiDim.y; i++) {
    rowsWidths_h[i] = dsiDim.x;
    
    rowsStart_h[i][0] = 0;
    rowsStart_h[i][1] = i;
    
    rowsEnd_h[i][0] = dsiDim.x - 1;
    rowsEnd_h[i][1] = i;    
  }
  
  cuerr << cudaThreadSynchronize();
  
  cuerr << cudaMemcpyToSymbol(rowsWidths, rowsWidths_h, sizeof(int)*MAX_ROWS);
  cuerr << cudaMemcpyToSymbol(rowsStart, rowsStart_h, sizeof(int)*MAX_ROWS*2);
  cuerr << cudaMemcpyToSymbol(rowsEnd, rowsEnd_h, sizeof(int)*MAX_ROWS*2);

  semiglobalAggregVolKernel<<<dsiDim.y, dsiDim.z>>>(dsiDim, dsi, lorigin,
                                                    1, 0, aggregDSI);    

  for (size_t i=0; i<dsiDim.x; i++) {
    rowsWidths_h[i] = dsiDim.y;

    rowsStart_h[i][0] = i;
    rowsStart_h[i][1] = 0;
    
    rowsEnd_h[i][0] = i;
    rowsEnd_h[i][1] = dsiDim.y - 1;    
  }
  
  cuerr << cudaThreadSynchronize();
  
  cuerr << cudaMemcpyToSymbol(rowsWidths, rowsWidths_h, sizeof(int)*MAX_ROWS);
  cuerr << cudaMemcpyToSymbol(rowsStart, rowsStart_h, sizeof(int)*MAX_ROWS*2);
  cuerr << cudaMemcpyToSymbol(rowsEnd, rowsEnd_h, sizeof(int)*MAX_ROWS*2);

  semiglobalAggregVolKernel<<<dsiDim.x, dsiDim.z>>>(dsiDim, dsi, lorigin,
                                                    0, 1, aggregDSI);
  const size_t lastX = dsiDim.x - 1;
  const size_t lastY = dsiDim.y - 1;
  
  for (size_t i=0; i<dsiDim.x; i++) {        
    rowsStart_h[i][0] = lastX - i;
    rowsStart_h[i][1] = 0;
    
    rowsEnd_h[i][0] = lastX;
    rowsEnd_h[i][1] = std::min(i, lastY);   
    
    rowsWidths_h[i] = rowsEnd_h[i][1] - rowsStart_h[i][1] + 1;
  }
  
  for (size_t i=0; i<dsiDim.y - 1; i++) {    
    const size_t offset = dsiDim.x + i;
    
    rowsStart_h[offset][0] = 0;
    rowsStart_h[offset][1] = i + 1;    
    
    rowsEnd_h[offset][0] = std::min(lastY - (i + 1), lastX);
    rowsEnd_h[offset][1] = lastY;

    rowsWidths_h[offset] = rowsEnd_h[offset][1] - rowsStart_h[offset][1] + 1;
  }
  

  cuerr << cudaThreadSynchronize();
  
  cuerr << cudaMemcpyToSymbol(rowsWidths, rowsWidths_h, sizeof(int)*MAX_ROWS);
  cuerr << cudaMemcpyToSymbol(rowsStart, rowsStart_h, sizeof(int)*MAX_ROWS*2);
  cuerr << cudaMemcpyToSymbol(rowsEnd, rowsEnd_h, sizeof(int)*MAX_ROWS*2);

  semiglobalAggregVolKernel<<<dsiDim.x + dsiDim.y - 1, dsiDim.z>>>(dsiDim, dsi, lorigin,
                                                                   1, 1, aggregDSI);  
  
  for (size_t i=0; i<dsiDim.x; i++) {         
    const int sX = i;
    const int sY = 0;
    const int diagSize = diagRSizeX(dsiDim.y, i);
    
    rowsStart_h[i][0] = sX;
    rowsStart_h[i][1] = sY;
    
    rowsWidths_h[i] = diagSize;
      
    rowsEnd_h[i][0] = 0;
    rowsEnd_h[i][1] = diagSize - 1;
  }
  
  for (size_t i=0; i<dsiDim.y - 1; i++) {        
    const size_t offset = dsiDim.x + i;
    
    const int sX = lastX;
    const int sY = i + 1;
    const int diagSize = diagRSizeY(dsiDim.x, dsiDim.y, i + 1);
    
    rowsStart_h[offset][0] = sX;
    rowsStart_h[offset][1] = sY;

    rowsEnd_h[offset][0] = sX - diagSize + 1;
    rowsEnd_h[offset][1] = sY + diagSize - 1;
    
    rowsWidths_h[offset] = diagSize;                    
  }
  
  cuerr << cudaThreadSynchronize();  
  
  cuerr << cudaMemcpyToSymbol(rowsWidths, rowsWidths_h, sizeof(int)*MAX_ROWS);
  cuerr << cudaMemcpyToSymbol(rowsStart, rowsStart_h, sizeof(int)*MAX_ROWS*2);

  cuerr << cudaMemcpyToSymbol(rowsEnd, rowsEnd_h, sizeof(int)*MAX_ROWS*2);
  
  semiglobalAggregVolKernel<<<dsiDim.x + dsiDim.y - 1, dsiDim.z>>>(dsiDim, dsi, lorigin,
                                                                   -1, 1, aggregDSI);

  cuerr << cudaThreadSynchronize();      

  wtaKernel<<<wsz.blocks, wsz.threads>>>(dsiDim, aggregDSI,
                                         dsiDim.x,
                                         dsiDim.x*dsiDim.y,
                                         dispImg);
}
