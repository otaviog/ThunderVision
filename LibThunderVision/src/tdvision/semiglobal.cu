#include <math_constants.h>
#include "cuerr.hpp"
#include "dim.hpp"
#include "dsimemutil.h"
#include "cudaconstraits.hpp"
#include "semiglobal.h"

//#define SG_USE_ATOMIC
#define SG_USE_P2ADJUST

__global__ void zeroDSIVolume(const dim3 dsiDim,
                              cudaPitchedPtr dsiVol)
{
  const uint x = blockDim.x*blockIdx.x + threadIdx.x;
  const uint y = blockDim.y*blockIdx.y + threadIdx.y;

  if ( x < dsiDim.x && y < dsiDim.y ) {
    float *dsiRow = dsiGetRow(dsiVol, dsiDim.y, x, y);
    //#pragma unroll 32
    for (uint z=0; z<dsiDim.z; z++) {
      dsiRow[z] = 0.0f;
    }
  }
}

__global__ void zeroDSIVolume2(const dim3 dsiDim,
                              cudaPitchedPtr dsiVol)
{
  const uint z = threadIdx.x;
  const uint y = blockIdx.x;
  __shared__ float *dsiRow;
  #pragma unroll 32
  for (uint x=0; x<dsiDim.x; x++) {
    if ( z == 0 ) {
      dsiRow = (float*) (((char*) dsiVol.ptr) + 
                         DSI_GET_ROW_INCR(dsiVol, dsiDim, x, y));
    }
    
    __syncthreads();    
    dsiRow[z] = 0.0f;
  }      
}

#define SG_MAX_DISP 256

TDV_NAMESPACE_BEGIN

void showDiagImg(int w, int h, SGPath *path, size_t pathCount);

SGPaths::SGPaths()
: m_imgDim(0)
{
  m_paths_d = NULL;
  m_pathCount = 0;
}

SGPath* SGPaths::getDescCPU(const Dim &imgDim, size_t *pathCountRet)
{
  const size_t w = imgDim.width();
  const size_t h = imgDim.height();

  size_t pathCount = w + h + (w + h - 1)*2;

  SGPath *paths = new SGPath[pathCount];

  SGPath *cpaths = paths;
  horizontalPathsDesc(imgDim, cpaths);

  cpaths = &cpaths[h];
  verticalPathsDesc(imgDim, cpaths);

  cpaths = &cpaths[w];
  bottomTopDiagonalDesc(imgDim, cpaths);

  cpaths = &cpaths[w + h - 1];
  topBottomDiagonaDesc(imgDim, cpaths);
  
  *pathCountRet = pathCount;
  
  return paths;
}

SGPath* SGPaths::getDescDev(const Dim &imgDim)
{
  if ( imgDim == m_imgDim )
    return m_paths_d;

  unalloc();

  m_imgDim = imgDim;

  SGPath *paths = getDescCPU(imgDim, &m_pathCount);
  
  CUerrExp cuerr;
  cuerr << cudaMalloc((void**) &m_paths_d, sizeof(SGPath)*m_pathCount);

  cuerr = cudaMemcpy(m_paths_d, paths, sizeof(SGPath)*m_pathCount,
                     cudaMemcpyHostToDevice);
  delete [] paths;

  cuerr.checkErr();

  return m_paths_d;
}

void SGPaths::unalloc()
{
  CUerrExp cuerr;

  cuerr << cudaFree(m_paths_d);

  m_paths_d = NULL;
  m_imgDim = Dim(0);
}

void SGPaths::horizontalPathsDesc(const Dim &imgDim, SGPath *paths)
{
  for (size_t i=0; i<imgDim.height(); i++) {
    paths[i].size = imgDim.width();

    paths[i].start.x = 0;
    paths[i].start.y = i;

    paths[i].end.x = imgDim.width() - 1;
    paths[i].end.y = i;

    paths[i].dir.x = 1;
    paths[i].dir.y = 0;
  }
}

void SGPaths::verticalPathsDesc(const Dim &imgDim, SGPath *paths)
{
  for (size_t i=0; i<imgDim.width(); i++) {
    paths[i].size = imgDim.height();

    paths[i].start.x = i;
    paths[i].start.y = 0;

    paths[i].end.x = i;
    paths[i].end.y = imgDim.height() - 1;

    paths[i].dir.x = 0;
    paths[i].dir.y = 1;
  }
}

/**
 *   /
 *  /
 * /
 *
 */
void SGPaths::bottomTopDiagonalDesc(const Dim &imgDim, SGPath *paths)
{
  const size_t lastX = imgDim.width() - 1;
  const size_t lastY = imgDim.height() - 1;

  for (size_t i=0; i<imgDim.width(); i++) {
    paths[i].start.x = lastX - i;
    paths[i].start.y = 0;

    paths[i].end.x = lastX;
    paths[i].end.y = std::min(i, lastY);

    paths[i].size = paths[i].end.y - paths[i].start.y + 1;
    paths[i].dir.x = 1;
    paths[i].dir.y = 1;
  }

  for (size_t i=0; i<imgDim.height() - 1; i++) {
    const size_t offset = imgDim.width() + i;

    paths[offset].start.x = 0;
    paths[offset].start.y = i + 1;

    paths[offset].end.x = std::min(lastY - (i + 1), lastX);
    paths[offset].end.y = lastY;

    paths[offset].size = paths[offset].end.x - paths[offset].start.x + 1;

    paths[offset].dir.x = 1;
    paths[offset].dir.y = 1;
  }
}

/**
 * \
 *  \
 */
void SGPaths::topBottomDiagonaDesc(const Dim &imgDim, SGPath *paths)
{
  const size_t lastX = imgDim.width() - 1;
  for (size_t i=0; i<imgDim.width(); i++) {
    const int sX = i;
    const int sY = 0;
    const int diagSize = std::min((int) i + 1, (int) imgDim.height());

    paths[i].start.x = sX;
    paths[i].start.y = sY;

    paths[i].size = diagSize;

    paths[i].end.x = 0;
    paths[i].end.y = diagSize - 1;

    paths[i].dir.x = -1;
    paths[i].dir.y = 1;
  }

  for (size_t i=0; i<imgDim.height() - 1; i++) {
    const size_t offset = imgDim.width() + i;

    const int sX = lastX;
    const int sY = i + 1;
    const int diagSize = std::min(static_cast<int>(((int) imgDim.height()) - (i + 1)),
                                  static_cast<int>(imgDim.width()));

    paths[offset].start.x = sX;
    paths[offset].start.y = sY;

    paths[offset].end.x = sX - diagSize + 1;
    paths[offset].end.y = sY + diagSize - 1;

    paths[offset].size = diagSize;

    paths[offset].dir.x = -1;
    paths[offset].dir.y = 1;
  }
}

TDV_NAMESPACE_END

struct __align__(8) fPTuple
{
  float forward, backward;
};

struct __align__(8) fptrPTuple
{
  float *forward, *backward;
};

struct __align__(8) fcptrPTuple
{
  const float *forward, *backward;
};

struct __align__(8) uPTuple
{
  uint forward, backward;
};

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
#if 1
  return cost + fmin4f(lcD,
                       lcDm1 + P1,
                       lcDp1 + P1,
                       minDisp + P2Adjust) - minDisp;
#else
  return __fadd_rn(__fadd_rn(cost, fmin4f(lcD,
                                          lcDm1,
                                          lcDp1,
                                          minDisp)),
                   -minDisp);
#endif
}

texture<float, 3> texCost;

template<int WarpThread>
__global__ void semiglobalAggregVolKernel(const dim3 dsiDim,
                                          const SGPath *pathsArray,
                                          const cudaPitchedPtr costDSI,
                                          const float *iImage,
                                          cudaPitchedPtr aggregDSI)
{
  const int z = threadIdx.x;
  const int y = blockIdx.x;
  const int dz = z + 1;

  __shared__ float sPrevCostF[SG_MAX_DISP + 2];
  __shared__ float sPrevMinCostF[SG_MAX_DISP];  
  __shared__ float sMinCostF;
  __shared__ int2 forwardCPt;
  __shared__ float* sCostDSIRowsPtrF;
  __shared__ float* sAggregDSIRowsPtrF;
  
  __shared__ float sPrevCostB[SG_MAX_DISP + 2];    
  __shared__ float sPrevMinCostB[SG_MAX_DISP];
  __shared__ float sMinCostB;  
  __shared__ int2 backwardCPt;
  __shared__ float* sCostDSIRowsPtrB;
  __shared__ float* sAggregDSIRowsPtrB;
  
#ifdef SG_USE_P2ADJUST
  __shared__ fPTuple sP2Adjust;
  __shared__ fPTuple sLastIntensity;
#endif        
    
  const SGPath *path = &pathsArray[y];
  const int pathSize = path->size;
  const int2 dir = { path->dir.x, path->dir.y };

  if ( z == 0 ) {
    forwardCPt.x = path->start.x;
    forwardCPt.y = path->start.y;
    
    sPrevCostF[0] = sPrevCostB[0] = CUDART_INF_F;
    
#ifdef SG_USE_P2ADJUST
    sLastIntensity.forward =
      iImage[forwardCPt.y*dsiDim.x + forwardCPt.x];
#endif
    
    sCostDSIRowsPtrF = dsiGetRow(costDSI, dsiDim.y,
                                 forwardCPt.x, forwardCPt.y);
    sAggregDSIRowsPtrF = dsiGetRow(aggregDSI, dsiDim.y, forwardCPt.x,
                                   forwardCPt.y);
  } else if ( z == WarpThread ) {
    backwardCPt.x = path->end.x;
    backwardCPt.y = path->end.y;
    
    sPrevCostF[dsiDim.z + 1] = sPrevCostB[dsiDim.z + 1] =
      CUDART_INF_F;
    
#ifdef SG_USE_P2ADJUST
    sLastIntensity.backward =
      iImage[backwardCPt.y*dsiDim.x + backwardCPt.x];
#endif

    sCostDSIRowsPtrB = dsiGetRow(costDSI, dsiDim.y, backwardCPt.x,
                                 backwardCPt.y);
    sAggregDSIRowsPtrB = dsiGetRow(aggregDSI, dsiDim.y, backwardCPt.x,
                                   backwardCPt.y);
  }    
  __syncthreads();

  float initialCost = sCostDSIRowsPtrF[z];

  sPrevCostF[dz] = initialCost + P1;
  sPrevMinCostF[z] = initialCost;
  sAggregDSIRowsPtrF[z] += initialCost;

  initialCost = sCostDSIRowsPtrB[z];
  sPrevCostB[dz] = initialCost + P1;
  sPrevMinCostB[z] = initialCost;
  
  sAggregDSIRowsPtrB[z] += initialCost;

  __syncthreads();

  float fLr, bLr;
  ushort dimz = dsiDim.z;

#pragma unroll 4
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

#ifdef SG_USE_P2ADJUST
      const float fI = iImage[forwardCPt.y*dsiDim.x + forwardCPt.x];

      sP2Adjust.forward = P2/abs(fI - sLastIntensity.forward);
      sLastIntensity.forward = fI;
#endif

      const size_t incr = DSI_GET_ROW_INCR(costDSI, dsiDim, 
                                           forwardCPt.x, forwardCPt.y);

      sCostDSIRowsPtrF = (float*) (((char*) costDSI.ptr) + incr);
      sAggregDSIRowsPtrF = (float*) (((char*) aggregDSI.ptr) + incr);

    } else if ( z == WarpThread ) {
      sMinCostB = sPrevMinCostB[dimz - 1];
      
      backwardCPt.x -= dir.x;
      backwardCPt.y -= dir.y;     
      
#ifdef SG_USE_P2ADJUST
      const float iI = iImage[backwardCPt.y*dsiDim.x + backwardCPt.x];

      sP2Adjust.backward = P2/abs(iI - sLastIntensity.backward);
      sLastIntensity.backward = iI;
#endif
      const size_t incr = DSI_GET_ROW_INCR(costDSI, dsiDim, 
                                           backwardCPt.x, backwardCPt.y);

      sCostDSIRowsPtrB = (float*) (((char*) costDSI.ptr) + incr);
      sAggregDSIRowsPtrB = (float*) (((char*) aggregDSI.ptr) + incr);
    }
    
    __syncthreads();

    fLr = pathCost(sCostDSIRowsPtrF[z],
                   //tex3D(texCost, forwardCPt.x, forwardCPt.y, z),
                   sPrevCostF[dz],
                   sPrevCostF[dz - 1],
                   sPrevCostF[dz + 1],
                   sMinCostF,
#ifdef SG_USE_P2ADJUST
                   sP2Adjust.forward
#else
                   P2
#endif
                   );

    bLr = pathCost(sCostDSIRowsPtrB[z],
                   //tex3D(texCost, backwardCPt.x, backwardCPt.y, z),
                   sPrevCostB[dz],
                   sPrevCostB[dz - 1],
                   sPrevCostB[dz + 1],
                   sMinCostB,
#ifdef SG_USE_P2ADJUST
                   sP2Adjust.backward
#else
                   P2
#endif
                   );

    __syncthreads();

    sPrevCostF[dz] = fLr;
    sPrevMinCostF[z] = fLr;
    sPrevCostB[dz] = bLr;
    sPrevMinCostB[z] = bLr;

#if 1
#ifdef SG_USE_ATOMIC
    atomicAdd(&sAggregDSIRowsPtrF[z], fLr);
    atomicAdd(&sAggregDSIRowsPtrB[z], bLr);
#else
    sAggregDSIRowsPtrF[z] += fLr;
    sAggregDSIRowsPtrB[z] += bLr;
#endif
#endif
    
    __syncthreads();
  }
}


TDV_NAMESPACE_BEGIN

void WTARunDev(const tdv::Dim &dsiDim, cudaPitchedPtr dsiMem, float *outimg);


void SemiGlobalDevRun(const Dim &dsiDim, cudaPitchedPtr dsi,
                      const SGPath *pathsArray, size_t pathCount,
                      const float *lorigin, cudaPitchedPtr aggregDSI,
                      float *dispImg, bool zeroAggregDSI)
{
  CUerrExp cuerr;
  CudaConstraits constraits;

  WorkSize wsz = constraits.imageWorkSize(dsiDim);
  
  //zeroAggregDSI = false;
  if ( zeroAggregDSI ) {
#if 0
    zeroDSIVolume<<<wsz.blocks, wsz.threads>>>(tdvDimTo(dsiDim), aggregDSI);
#else
    
    zeroDSIVolume2<<<dsiDim.height(), 
      dsiDim.depth()>>>(tdvDimTo(dsiDim), aggregDSI);
#endif
  }
  
  texCost.addressMode[0] = cudaAddressModeWrap;
  texCost.addressMode[1] = cudaAddressModeWrap;
  texCost.addressMode[2] = cudaAddressModeWrap;
  texCost.normalized = false;
  texCost.filterMode = cudaFilterModePoint;
  
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  
  cuerr << cudaBindTexture(NULL, texCost, dsi.ptr, 
                           desc,
                           dsi.pitch*dsiDim.width()*dsiDim.height());    
  
  if ( dsiDim.depth() > 32 )
    semiglobalAggregVolKernel<32>
      <<<pathCount, dsiDim.depth()>>>(tdvDimTo(dsiDim), pathsArray,
                                      dsi, lorigin,
                                      aggregDSI);
  else
    semiglobalAggregVolKernel<1>
      <<<pathCount, dsiDim.depth()>>>(tdvDimTo(dsiDim), pathsArray,
                                      dsi, lorigin,
                                      aggregDSI);

  WTARunDev(dsiDim, aggregDSI, dispImg);

  cuerr << cudaThreadSynchronize();
}

TDV_NAMESPACE_END