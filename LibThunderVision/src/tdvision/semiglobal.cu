#include <math_constants.h>
#include "cuerr.hpp"
#include "dim.hpp"
#include "dsimemutil.h"
#include "cudaconstraits.hpp"
#include "semiglobal.h"

__global__ void wtaKernel(const dim3 dim, 
                          cudaPitchedPtr dsiMem,
                          float *outimg);

__global__ void zeroDSIVolume(const dim3 dsiDim,
                              cudaPitchedPtr dsiVol)
{
  const uint x = blockDim.x*blockIdx.x + threadIdx.x;
  const uint y = blockDim.y*blockIdx.y + threadIdx.y;

  if ( x < dsiDim.x && y < dsiDim.y ) {
    float *dsiRow = dsiGetRow(dsiVol, dsiDim.x, x, y);
    
    for (uint z=0; z<dsiDim.z; z++) {
      dsiRow[z] = 0.0f;
    }
  }
}

#define SG_MAX_DISP 512
#define SG_WARP_SIZE 32

TDV_NAMESPACE_BEGIN

void showDiagImg(int w, int h, SGPath *path, size_t pathCount);

SGPaths::SGPaths()
: m_imgDim(0)
{
  m_paths_d = NULL;
  m_pathCount = 0;
}

SGPath* SGPaths::getDesc(const Dim &imgDim)
{
  if ( imgDim == m_imgDim )
    return m_paths_d;

  unalloc();

  m_imgDim = imgDim;
  
  const size_t w = imgDim.width();
  const size_t h = imgDim.height();
  
  m_pathCount = w + h + (w + h - 1)*2;
  
  SGPath *paths = new SGPath[m_pathCount];

  SGPath *cpaths = paths;  
  horizontalPathsDesc(cpaths);                

  cpaths = &cpaths[h];  
  verticalPathsDesc(cpaths);
  
  cpaths = &cpaths[w];
  bottomTopDiagonalDesc(cpaths);  
              
  cpaths = &cpaths[w + h - 1];
  topBottomDiagonaDesc(cpaths);
  
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

void SGPaths::horizontalPathsDesc(SGPath *paths)
{
  for (size_t i=0; i<m_imgDim.height(); i++) {
    paths[i].size = m_imgDim.width();

    paths[i].start.x = 0;
    paths[i].start.y = i;

    paths[i].end.x = m_imgDim.width() - 1;
    paths[i].end.y = i;
    
    paths[i].dir.x = 1;
    paths[i].dir.y = 0;
  }
}

void SGPaths::verticalPathsDesc(SGPath *paths)
{
  for (size_t i=0; i<m_imgDim.width(); i++) {
    paths[i].size = m_imgDim.height();

    paths[i].start.x = i;
    paths[i].start.y = 0;

    paths[i].end.x = i;
    paths[i].end.y = m_imgDim.height() - 1;

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
void SGPaths::bottomTopDiagonalDesc(SGPath *paths)
{
  const size_t lastX = m_imgDim.width() - 1;
  const size_t lastY = m_imgDim.height() - 1;

  for (size_t i=0; i<m_imgDim.width(); i++) {
    paths[i].start.x = lastX - i;
    paths[i].start.y = 0;

    paths[i].end.x = lastX;
    paths[i].end.y = std::min(i, lastY);

    paths[i].size = paths[i].end.y - paths[i].start.y + 1;
    paths[i].dir.x = 1;
    paths[i].dir.y = 1;
  }

  for (size_t i=0; i<m_imgDim.height() - 1; i++) {
    const size_t offset = m_imgDim.width() + i;

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

void SGPaths::topBottomDiagonaDesc(SGPath *paths)
{
  const size_t lastX = m_imgDim.width() - 1;
  for (size_t i=0; i<m_imgDim.width(); i++) {
    const int sX = i;
    const int sY = 0;
    const int diagSize = std::min((int) i + 1, (int) m_imgDim.height());

    paths[i].start.x = sX;
    paths[i].start.y = sY;

    paths[i].size = diagSize;

    paths[i].end.x = 0;
    paths[i].end.y = diagSize - 1;
    
    paths[i].dir.x = -1;
    paths[i].dir.y = 1;
  }

  for (size_t i=0; i<m_imgDim.height() - 1; i++) {
    const size_t offset = m_imgDim.width() + i;

    const int sX = lastX;
    const int sY = i + 1;
    const int diagSize = std::min(static_cast<int>(((int) m_imgDim.height()) - (i + 1)),
                                  static_cast<int>(m_imgDim.width()));

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

struct fPTuple
{
  float forward, backward;
};

struct fptrPTuple
{
  float *forward, *backward;
};

struct fcptrPTuple
{
  const float *forward, *backward;
};

struct uPTuple
{
  uint forward, backward;
};

const float P1 = 40.0f/255.0f;
const float P2 = 180.0f/255.0f;

#define min4(a, b, c, d) min(a, min(b, min(c, d)))

using tdv::SGPoint;
using tdv::SGPath;
using tdv::sSGPoint;

inline __device__ float pathCost(float cost, 
                                 float lcD, float lcDm1, float lcDp1,
                                 float minDisp, float P2Adjust)
{
  const float Lr = cost
    + min4(lcD,
           lcDm1 + P1,
           lcDp1 + P1,
           minDisp + P2Adjust) - minDisp;
  
  return Lr;
}

__global__ void semiglobalAggregVolKernel(const dim3 dsiDim,
                                          const SGPath *pathsArray,
                                          const cudaPitchedPtr costDSI,
                                          const float *iImage,
                                          cudaPitchedPtr aggregDSI)
{
  const ushort z = threadIdx.x;
  const ushort y = blockIdx.x;
  const ushort dz = z + 1;

  __shared__ fPTuple sPrevCost[SG_MAX_DISP + 2];
  __shared__ fPTuple sPrevMinCost[SG_MAX_DISP];
  __shared__ fPTuple sMinCost;
  __shared__ fPTuple sP2Adjust;
  __shared__ fPTuple sLastIntensity;
  __shared__ SGPoint forwardCPt, backwardCPt;
  __shared__ fcptrPTuple sCostDSIRowsPtr;
  __shared__ fptrPTuple  sAggregDSIRowsPtr;
  
  const SGPath *path = &pathsArray[y];
  const ushort pathSize = path->size;
  const sSGPoint dir = path->dir;
    
  if ( z == 0 ) {          
    forwardCPt = path->start;    
    backwardCPt = path->end;    
      
    sPrevCost[0].forward = sPrevCost[0].backward = CUDART_INF_F;
    sPrevCost[dsiDim.z + 1].forward = sPrevCost[dsiDim.z + 1].backward = 
      CUDART_INF_F;

    sLastIntensity.forward = 
      iImage[forwardCPt.y*dsiDim.x + forwardCPt.x];
      
    sLastIntensity.backward =
      iImage[backwardCPt.y*dsiDim.x + backwardCPt.x];

  }

  __syncthreads();
      
  const float *dsiRow = dsiGetRow(costDSI, dsiDim.x, 
                                  forwardCPt.x, forwardCPt.y);
  float initialCost = dsiRow[z];

  sPrevCost[dz].forward = initialCost;
  sPrevMinCost[z].forward = initialCost;
  
  float *dsiAgrregRow = dsiGetRow(aggregDSI, dsiDim.x, forwardCPt.x, 
                     forwardCPt.y);
  dsiAgrregRow[z] += initialCost;
      
  dsiRow = dsiGetRow(costDSI, dsiDim.x, backwardCPt.x, 
                     backwardCPt.y);
  initialCost = dsiRow[z];
      
  sPrevCost[dz].backward = initialCost;
  sPrevMinCost[z].backward = initialCost;
      
  dsiAgrregRow = dsiGetRow(aggregDSI, dsiDim.x, backwardCPt.x, 
                     backwardCPt.y);
  dsiAgrregRow[z] += initialCost;

  __syncthreads();

  float fLr, bLr;
  ushort dimz = dsiDim.z;
  
  for (ushort x=1; x<pathSize; x++) {
    ushort i = dimz >> 1;
    while ( i != 0 ) {
      if ( z < i ) {
        sPrevMinCost[z].forward = min(sPrevMinCost[z].forward, 
                                      sPrevMinCost[z + i].forward);
      } else if ( dimz - z <= i ) {
        sPrevMinCost[z].backward = min(sPrevMinCost[z].backward, 
                                       sPrevMinCost[z - i].backward);
      }
      __syncthreads();      
      i = i >> 1;
    }    
    
    if ( z == 0 ) {
      sMinCost.forward = sPrevMinCost[0].forward;
      forwardCPt.x += dir.x;
      forwardCPt.y += dir.y;

      const float fI = iImage[forwardCPt.y*dsiDim.x + forwardCPt.x];

      sP2Adjust.forward = P2/abs(fI - sLastIntensity.forward);      
      sLastIntensity.forward = fI;

      const size_t incr = DSI_GET_ROW_INCR(costDSI, dsiDim, forwardCPt.x, forwardCPt.y);
      
      sCostDSIRowsPtr.forward = (float*) (((char*) costDSI.ptr) + incr);
      sAggregDSIRowsPtr.forward = (float*) (((char*) aggregDSI.ptr) + incr);

  } else if ( z == SG_WARP_SIZE ) {
      sMinCost.backward = sPrevMinCost[dimz - 1].backward;

      backwardCPt.x -= dir.x;
      backwardCPt.y -= dir.y;

      const float iI = iImage[backwardCPt.y*dsiDim.x + backwardCPt.x];

      sP2Adjust.backward = P2/abs(iI - sLastIntensity.backward);

      sLastIntensity.backward = iI;
      
      const size_t incr = DSI_GET_ROW_INCR(costDSI, dsiDim, backwardCPt.x, backwardCPt.y);
      
      sCostDSIRowsPtr.backward = (float*) (((char*) costDSI.ptr) + incr);
      sAggregDSIRowsPtr.backward = (float*) (((char*) aggregDSI.ptr) + incr);      
  }
    
    __syncthreads();

    fLr = pathCost(sCostDSIRowsPtr.forward[z],
                   sPrevCost[dz].forward,
                   sPrevCost[dz - 1].forward,
                   sPrevCost[dz + 1].forward,
                   sMinCost.forward, sP2Adjust.forward);
        
    sAggregDSIRowsPtr.forward[z] += fLr;                        
    //atomicAdd(&sAggregDSIRowsPtr.forward[z], fLr);
    
    bLr = pathCost(sCostDSIRowsPtr.backward[z],
                   sPrevCost[dz].backward,
                   sPrevCost[dz - 1].backward,
                   sPrevCost[dz + 1].backward,
                   sMinCost.backward, sP2Adjust.backward);
    
    sAggregDSIRowsPtr.backward[z] += bLr;
    //atomicAdd(&sAggregDSIRowsPtr.backward[z], bLr);

    __syncthreads();
        
    sPrevCost[dz].forward = sPrevMinCost[z].forward = fLr;
    sPrevCost[dz].backward = sPrevMinCost[z].backward = bLr;

    __syncthreads();
  }
}


TDV_NAMESPACE_BEGIN

void WTARunDev(const tdv::Dim &dsiDim, cudaPitchedPtr dsiMem, float *outimg);

void SemiGlobalDevRun(const tdv::Dim &dsiDim, cudaPitchedPtr dsi,
                      const tdv::SGPath *pathsArray, size_t pathCount,
                      const float *lorigin, cudaPitchedPtr aggregDSI, 
                      float *dispImg, bool zeroAggregDSI)
{  
  tdv::CUerrExp cuerr;
  tdv::CudaConstraits constraits;

  tdv::WorkSize wsz = constraits.imageWorkSize(dsiDim);

  if ( zeroAggregDSI )
    zeroDSIVolume<<<wsz.blocks, wsz.threads>>>(tdvDimTo(dsiDim), aggregDSI);
  
  semiglobalAggregVolKernel
    <<<pathCount, dsiDim.depth()>>>(tdvDimTo(dsiDim), pathsArray,
                                    dsi, lorigin,
                                    aggregDSI);   

  WTARunDev(dsiDim, aggregDSI, dispImg);
  #if 0 
  wtaKernel<<<wsz.blocks, wsz.threads>>>(tdvDimTo(dsiDim), aggregDSI,
                                         dispImg);
#endif
  cuerr << cudaThreadSynchronize();
}

TDV_NAMESPACE_END