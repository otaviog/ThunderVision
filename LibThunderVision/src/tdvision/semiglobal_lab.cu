#include <math_constants.h>
#include <limits.h>
#include "cuerr.hpp"
#include "dim.hpp"
#include "dsimemutil.h"
#include "cudaconstraits.hpp"
#include "semiglobal.h"

//#define SG_USE_ATOMIC

__global__ void zeroDSIVolume(const dim3 dsiDim,
                              cudaPitchedPtr dsiVol);

__global__ void zeroDSIVolume2(const dim3 dsiDim,
                               cudaPitchedPtr dsiVol);

#define SG_MAX_DISP 256

const float P1 = 30.0f/255.0f;
const float P2 = 150.0f/255.0f;

#define fmin4f(a, b, c, d) fminf(a, fminf(b, fminf(c, d)))

using tdv::SGPoint;
using tdv::SGPath;
using tdv::sSGPoint;

inline __device__ float pathCost(float cost,
                                 float lcD, float lcDm1, float lcDp1,
                                 float minDisp, float P2Adjust)
{
  return cost + fmin4f(lcD,
                       lcDm1 + P1,
                       lcDp1 + P1,
                       minDisp + P2Adjust) - minDisp;
}

texture<float, 2> texLeftImage;
texture<float, 2> texRightImage;

#define min3(a, b, c) min(a, min(b, c))
#define max3(a, b, c) max(a, max(b, c))

#if 0
#define SSD_WIND_DIM 5
#define SSD_WIND_START -3
#define SSD_WIND_END 4

__device__ float costAtDisp(int x, int y, int disp)
{
  if ( x - disp < 0 )
    return CUDART_INF_F;
  
  float sum = 0.0f;

  for (int row=SSD_WIND_START; row<SSD_WIND_END; row++)
    for (int col=SSD_WIND_START; col<SSD_WIND_END; col++) {

      float lI = tex2D(texLeftImage, x + col, y + row),
        rI = tex2D(texRightImage, x + col - disp, y + row);

      sum += (lI - rI)*(lI - rI);
    }

  return sum;
}
#else
__device__ short costAtDisp(int x, int y, int disp)
{
  float sum = 0.0f;
  if ( x - disp < 0 )
    return CUDART_INF_F;
 
  for (int v=x; v < x + 1; v++) {
    const float lI = tex2D(texLeftImage, v, y);
    const float rI = tex2D(texRightImage, v - disp, y);

    const float laI = 0.5f*(lI + tex2D(texLeftImage, v - 1, y));
    const float lbI = 0.5f*(lI + tex2D(texLeftImage, v + 1, y));

    const float raI = 0.5f*(rI + tex2D(texRightImage, v - disp - 1, y));
    const float rbI = 0.5f*(rI + tex2D(texRightImage, v - disp + 1, y));

    const float lImi = min3(laI, lbI, lI);
    const float lIma = max3(laI, lbI, lI);

    const float rImi = min3(raI, rbI, rI);
    const float rIma = max3(raI, rbI, rI);

    sum += min(max3(0.0f, lI - rIma, rImi - lI),
               max3(0.0f, rI - lIma, lImi - rI));
  }

  return sum;
}
#endif

template<int WarpThread>
__global__ void semiglobalLabAggregVolKernel(const dim3 dsiDim,
                                             const SGPath *pathsArray,
                                             cudaPitchedPtr aggregDSI)
{
  const int z = threadIdx.x;
  const int y = blockIdx.x;
  const int dz = z + 1;

  __shared__ float sPrevCostF[SG_MAX_DISP + 2];
  __shared__ float sPrevMinCostF[SG_MAX_DISP];  
  __shared__ float sMinCostF;
  __shared__ int2 forwardCPt;
  __shared__ float* sAggregDSIRowsPtrF;
  
  __shared__ float sPrevCostB[SG_MAX_DISP + 2];    
  __shared__ float sPrevMinCostB[SG_MAX_DISP];
  __shared__ float sMinCostB;  
  __shared__ int2 backwardCPt;
  __shared__ float* sAggregDSIRowsPtrB;
      
  const SGPath *path = &pathsArray[y];
  const int pathSize = path->size;
  const int2 dir = { path->dir.x, path->dir.y };

  if ( z == 0 ) {
    forwardCPt.x = path->start.x;
    forwardCPt.y = path->start.y;
    
    sPrevCostF[0] = sPrevCostB[0] = CUDART_INF_F;
        
    sAggregDSIRowsPtrF = dsiGetRow(aggregDSI, dsiDim.y, forwardCPt.x,
                                   forwardCPt.y);
  } else if ( z == WarpThread ) {
    backwardCPt.x = path->end.x;
    backwardCPt.y = path->end.y;
    
    sPrevCostF[dsiDim.z + 1] = sPrevCostB[dsiDim.z + 1] = CUDART_INF_F;
    
    sAggregDSIRowsPtrB = dsiGetRow(aggregDSI, dsiDim.y, 
                                   backwardCPt.x, backwardCPt.y);
  }    
  __syncthreads();

  float initialCost = costAtDisp(forwardCPt.x, forwardCPt.y, z);

  sPrevCostF[dz] = initialCost;
  sPrevMinCostF[z] = initialCost;
  sAggregDSIRowsPtrF[z] += initialCost;

  initialCost = costAtDisp(backwardCPt.x, backwardCPt.y, z);
  sPrevCostB[dz] = initialCost;
  sPrevMinCostB[z] = initialCost;
  
  sAggregDSIRowsPtrB[z] += initialCost;

  __syncthreads();

  float fLr, bLr;
  ushort dimz = dsiDim.z;

  for (int x=1; x<pathSize; x++) {
    int i = dimz >> 1;
    while ( i != 0 ) {
      if ( z < i ) {
        sPrevMinCostF[z] = fminf(sPrevMinCostF[z],
                                 sPrevMinCostF[z + i]);
      } else if ( dimz - z <= i ) {
        sPrevMinCostB[z] = fminf(sPrevMinCostB[z],
                                 sPrevMinCostB[z - i]);
      }
      __syncthreads();
      i = i >> 1;
    }
        
    if ( z == 0 ) {
      sMinCostF = sPrevMinCostF[0];

      forwardCPt.x += dir.x;
      forwardCPt.y += dir.y;

      sAggregDSIRowsPtrF = dsiGetRowT<float*>(aggregDSI, dsiDim, 
                                              forwardCPt.x, forwardCPt.y);     

    } else if ( z == WarpThread ) {
      sMinCostB = sPrevMinCostB[dimz - 1];
      
      backwardCPt.x -= dir.x;
      backwardCPt.y -= dir.y;     
      
      sAggregDSIRowsPtrB = dsiGetRowT<float*>(aggregDSI, dsiDim, 
                                              backwardCPt.x, backwardCPt.y);
    }
    
    __syncthreads();
    
    fLr = pathCost(costAtDisp(forwardCPt.x, forwardCPt.y, z),
                   sPrevCostF[dz],
                   sPrevCostF[dz - 1],
                   sPrevCostF[dz + 1],
                   sMinCostF,
                   P2);
    
    bLr = pathCost(costAtDisp(backwardCPt.x, backwardCPt.y, z),
                   sPrevCostB[dz],
                   sPrevCostB[dz - 1],
                   sPrevCostB[dz + 1],
                   sMinCostB,
                   P2);

    __syncthreads();

    sPrevCostF[dz] = fLr;
    sPrevMinCostF[z] = fLr;
    sPrevCostB[dz] = bLr;
    sPrevMinCostB[z] = bLr;

#ifdef SG_USE_ATOMIC
    atomicAdd(&sAggregDSIRowsPtrF[z], fLr);
    atomicAdd(&sAggregDSIRowsPtrB[z], bLr);
#else
    sAggregDSIRowsPtrF[z] += fLr;
    sAggregDSIRowsPtrB[z] += bLr;
#endif
    
    __syncthreads();
  }
}


TDV_NAMESPACE_BEGIN

void WTARunDev(const tdv::Dim &dsiDim, cudaPitchedPtr dsiMem, float *outimg);

void SemiGlobalLabDevRun(const Dim &dsiDim,
                         const SGPath *pathsArray, size_t pathCount,
                         float *leftImg_d, float *rightImg_d, 
                         cudaPitchedPtr aggregDSI,
                         float *dispImg, bool zeroAggregDSI)
{
  CUerrExp cuerr;
  CudaConstraits constraits;

  WorkSize wsz = constraits.imageWorkSize(dsiDim);
   
  if ( zeroAggregDSI ) {
#if 1
    zeroDSIVolume<<<wsz.blocks, wsz.threads>>>(tdvDimTo(dsiDim), aggregDSI);
#else
    
    zeroDSIVolume2<<<dsiDim.height(), 
      dsiDim.depth()>>>(tdvDimTo(dsiDim), aggregDSI);
#endif
  }
  
  cuerr << cudaBindTexture2D(NULL, texLeftImage, leftImg_d,
                           cudaCreateChannelDesc<float>(),
                           dsiDim.width(), dsiDim.height(),
                           dsiDim.width()*sizeof(float));

  cuerr << cudaBindTexture2D(NULL, texRightImage, rightImg_d,
                           cudaCreateChannelDesc<float>(),
                           dsiDim.width(), dsiDim.height(),
                           dsiDim.width()*sizeof(float));

  texLeftImage.addressMode[0] = texRightImage.addressMode[0] = 
    cudaAddressModeWrap;
  texLeftImage.addressMode[1] = texRightImage.addressMode[1] = 
    cudaAddressModeWrap;
  texLeftImage.normalized = texRightImage.normalized = false;
  texLeftImage.filterMode = texRightImage.filterMode = cudaFilterModePoint;
  
  if ( dsiDim.depth() > 32 )
    semiglobalLabAggregVolKernel<32>
      <<<pathCount, dsiDim.depth()>>>(tdvDimTo(dsiDim), pathsArray,
                                      aggregDSI);
  else
    semiglobalLabAggregVolKernel<1>
      <<<pathCount, dsiDim.depth()>>>(tdvDimTo(dsiDim), pathsArray,
                                      aggregDSI);

  WTARunDev(dsiDim, aggregDSI, dispImg);

  cuerr << cudaThreadSynchronize();
}

TDV_NAMESPACE_END