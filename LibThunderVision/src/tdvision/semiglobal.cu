#include <math_constants.h>
#include "dim.hpp"
#include "dsimemutil.h"
#include "cudaconstraits.hpp"
#include "semiglobal.h"


void showDiagImg(int w, int h, int *st, int *ed);

__global__ void wtaKernel(const DSIDim dim,
                          float *dsi,
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

#define SG_MAX_DISP 512

TDV_NAMESPACE_BEGIN

static void sg_rowsWidthsFill(int value, uint ntimes, int *rowsWidths_h)
{
  for (uint i=0; i<ntimes; i++) {
    rowsWidths_h[i] = value;
  }
}

SGPaths::SGPaths()
: m_imgDim(0)
{
  m_desc_d = NULL;
}

SGPathsDesc* SGPaths::getDesc(const Dim &imgDim)
{
  if ( imgDim == m_imgDim )
    return m_desc_d;

  unalloc();

  m_imgDim = imgDim;

  const uint maxPathsPerPass = m_imgDim.width() + m_imgDim.height() - 1;

  CUerrExp cuerr;
  cuerr << cudaMalloc((void**) &m_desc_d, sizeof(SGPathsDesc));

  SGPathsDesc *desc = new SGPathsDesc;

  horizontalPathsDesc(maxPathsPerPass, desc->pathsSizes[0],
                      desc->pathsStarts[0], desc->pathsEnds[0]);

  verticalPathsDesc(maxPathsPerPass, desc->pathsSizes[1],
                    desc->pathsStarts[1], desc->pathsEnds[1]);

  bottomTopDiagonalDesc(desc->pathsSizes[2], desc->pathsStarts[2],
                        desc->pathsEnds[2]);

  topBottomDiagonaDesc(desc->pathsSizes[3], desc->pathsStarts[3],
                       desc->pathsEnds[3]);


  for (size_t i=0; i<SG_MAX_PATHS; i++) {
    desc->maxPathsSizes[i] = desc->pathsSizes[0][i];

    for (size_t p=1; p<SG_MAX_PASSES; p++) {
      desc->maxPathsSizes[i] = std::max(desc->pathsSizes[p][i],
                                        desc->maxPathsSizes[i]);
    }
  }

  cuerr = cudaMemcpy(m_desc_d, desc, sizeof(SGPathsDesc),
                      cudaMemcpyHostToDevice);
  delete [] desc;

  cuerr.checkErr();

  return m_desc_d;
}

void SGPaths::unalloc()
{
  CUerrExp cuerr;

  cuerr << cudaFree(m_desc_d);

  m_desc_d = NULL;
  m_imgDim = Dim(0);
}

void SGPaths::horizontalPathsDesc(uint totalPaths, int *pathsSizes,
                                  SGPoint *pathsStarts, SGPoint *pathsEnds)
{
  for (size_t i=0; i<m_imgDim.height(); i++) {
    pathsSizes[i] = m_imgDim.width();

    pathsStarts[i].x = 0;
    pathsStarts[i].y = i;

    pathsEnds[i].x = m_imgDim.width() - 1;
    pathsEnds[i].y = i;
  }

  sg_rowsWidthsFill(0, SG_MAX_PATHS - m_imgDim.height(), &pathsSizes[m_imgDim.height()]);
}

void SGPaths::verticalPathsDesc(uint totalPaths, int *pathSizes, SGPoint *pathsStarts,
                                SGPoint *pathsEnds)
{
  for (size_t i=0; i<m_imgDim.width(); i++) {
    pathSizes[i] = m_imgDim.height();

    pathsStarts[i].x = i;
    pathsStarts[i].y = 0;

    pathsEnds[i].x = i;
    pathsEnds[i].y = m_imgDim.height() - 1;
  }

  sg_rowsWidthsFill(0, SG_MAX_PATHS - m_imgDim.width(), &pathSizes[m_imgDim.width()]);
}

/**
 *   /
 *  /
 * /
 *
 */
void SGPaths::bottomTopDiagonalDesc(int *pathSizes, SGPoint *pathsStarts, SGPoint *pathsEnds)
{
  const size_t lastX = m_imgDim.width() - 1;
  const size_t lastY = m_imgDim.height() - 1;

  for (size_t i=0; i<m_imgDim.width(); i++) {
    pathsStarts[i].x = lastX - i;
    pathsStarts[i].y = 0;

    pathsEnds[i].x = lastX;
    pathsEnds[i].y = std::min(i, lastY);

    pathSizes[i] = pathsEnds[i].y - pathsStarts[i].y + 1;
  }

  for (size_t i=0; i<m_imgDim.height() - 1; i++) {
    const size_t offset = m_imgDim.width() + i;

    pathsStarts[offset].x = 0;
    pathsStarts[offset].y = i + 1;

    pathsEnds[offset].x = std::min(lastY - (i + 1), lastX);
    pathsEnds[offset].y = lastY;

    pathSizes[offset] = pathsEnds[offset].x - pathsStarts[offset].x + 1;
  }
}

/**
 * \
 *  \
 */

void SGPaths::topBottomDiagonaDesc(int *pathSizes, SGPoint *pathsStarts, SGPoint *pathsEnds)
{
  const size_t lastX = m_imgDim.width() - 1;
  for (size_t i=0; i<m_imgDim.width(); i++) {
    const int sX = i;
    const int sY = 0;
    const int diagSize = std::min((int) i + 1, (int) m_imgDim.height());

    pathsStarts[i].x = sX;
    pathsStarts[i].y = sY;

    pathSizes[i] = diagSize;

    pathsEnds[i].x = 0;
    pathsEnds[i].y = diagSize - 1;
  }

  for (size_t i=0; i<m_imgDim.height() - 1; i++) {
    const size_t offset = m_imgDim.width() + i;

    const int sX = lastX;
    const int sY = i + 1;
    const int diagSize = std::min(static_cast<int>(((int) m_imgDim.height()) - (i + 1)),
                                  static_cast<int>(m_imgDim.width()));

    pathsStarts[offset].x = sX;
    pathsStarts[offset].y = sY;

    pathsEnds[offset].x = sX - diagSize + 1;
    pathsEnds[offset].y = sY + diagSize - 1;

    pathSizes[offset] = diagSize;
  }
}

TDV_NAMESPACE_END

struct fPTuple
{
  float forward, backward;
};

struct uPTuple
{
  uint forward, backward;
};

#define N_PASSES 2

const float P1 = 40.0f/255.0f;
const float P2 = 180.0f/255.0f;

#define min4(a, b, c, d) min(a, min(b, min(c, d)))

using tdv::SGPoint;
using tdv::SGPathsDesc;

__constant__ SGPoint g_pathsDirections_d[SG_MAX_PASSES];

__device__ float pathCost(const float *costDSI, uint dsiOff,
                          float lcD, float lcDm1, float lcDp1,
                          float minDisp, float P2Adjust,
                          float *aggregVol)
{
  const float cost = 
    costDSI[dsiOff];

  const float Lr = cost
    + min4(lcD,
           lcDm1 + P1,
           lcDp1 + P1,
           minDisp + P2Adjust) - minDisp;

  aggregVol[dsiOff] += Lr;

  return Lr;
}

__global__ void semiglobalAggregVolKernel(const DSIDim dsiDim,
                                          const SGPathsDesc *pathsDesc,
                                          const float *costDSI,
                                          const float *iImage,
                                          float *aggregDSI)
{
  const ushort z = threadIdx.x;
  const ushort y = blockIdx.x;
  const ushort dz = z + 1;

  __shared__ fPTuple sPrevCost[SG_MAX_PASSES][SG_MAX_DISP + 2];
  __shared__ fPTuple sPrevMinCost[SG_MAX_PASSES][SG_MAX_DISP];
  __shared__ fPTuple sMinCost[SG_MAX_PASSES];
  __shared__ fPTuple sP2Adjust[SG_MAX_PASSES];
  __shared__ fPTuple sLastIntensity[SG_MAX_PASSES];
  __shared__ SGPoint forwardCPt[SG_MAX_PASSES], backwardCPt[SG_MAX_PASSES];
  __shared__ uPTuple sDsiBaseOff[SG_MAX_PASSES];  
  __shared__ ushort pathSizes[SG_MAX_PASSES];
  
  const ushort maxPathSize = pathsDesc->maxPathsSizes[y];
  if ( z < N_PASSES ) {
    const short p = z;
    
    pathSizes[z] = pathsDesc->pathsSizes[z][y];
    
    if ( pathSizes[z] > 0 ) {
      
      forwardCPt[p].x = pathsDesc->pathsStarts[p][y].x;
      forwardCPt[p].y = pathsDesc->pathsStarts[p][y].y;

      backwardCPt[p].x = pathsDesc->pathsEnds[p][y].x;
      backwardCPt[p].y = pathsDesc->pathsEnds[p][y].y;
      
      sPrevCost[p][0].forward = sPrevCost[p][0].backward = CUDART_INF_F;
      sPrevCost[p][dsiDim.z + 1].forward =
      sPrevCost[p][dsiDim.z + 1].backward = CUDART_INF_F;
      
      sLastIntensity[p].forward = 
        iImage[forwardCPt[p].y*dsiDim.x + forwardCPt[p].x];
      sLastIntensity[p].backward =
        iImage[backwardCPt[p].y*dsiDim.x + backwardCPt[p].x];
    }
  }

  __syncthreads();

  for (char p=0; p<N_PASSES; p++) {
    if ( pathSizes[p] > 0 ) {
      uint initialOff = dsiOffset(dsiDim, forwardCPt[p].x, forwardCPt[p].y, z);
      float initialCost = 
        costDSI[initialOff];

      sPrevCost[p][dz].forward = initialCost;
      sPrevMinCost[p][z].forward = initialCost;
      aggregDSI[initialOff] += initialCost;

      initialOff = dsiOffset(dsiDim, backwardCPt[p].x, backwardCPt[p].y, z);
      initialCost = 
        costDSI[initialOff];

      sPrevCost[p][dz].backward = initialCost;
      sPrevMinCost[p][z].backward = initialCost;
      aggregDSI[initialOff] += initialCost;
    }
  }

  __syncthreads();

  float fLr[N_PASSES], bLr[N_PASSES];
  for (uint x=1; x<maxPathSize; x++) {
    ushort i = dsiDim.z >> 1;
    while ( i != 0 ) {
      if ( z < i ) {
        for (char p=0; p<N_PASSES; p++) {
          if ( x >= pathSizes[p] )
            continue;

          sPrevMinCost[p][z].forward = min(sPrevMinCost[p][z].forward,
                                           sPrevMinCost[p][z + i].forward);
          sPrevMinCost[p][z].backward = min(sPrevMinCost[p][z].backward,
                                            sPrevMinCost[p][z + i].backward);
        }
      }

      __syncthreads();
      i = i >> 1;
    }

    if ( z < N_PASSES && x < pathSizes[z] ) {
      const char p = z;

      sMinCost[p].forward = sPrevMinCost[p][0].forward;

      forwardCPt[p].x += g_pathsDirections_d[p].x;
      forwardCPt[p].y += g_pathsDirections_d[p].y;

      const float fI = iImage[forwardCPt[p].y*dsiDim.x + forwardCPt[p].x];

      sP2Adjust[p].forward = P2/abs(fI - sLastIntensity[p].forward);

      sLastIntensity[p].forward = fI;

      sDsiBaseOff[p].forward = dsiDim.z*dsiDim.y*forwardCPt[p].x
        + dsiDim.z*forwardCPt[p].y;
    }
    else if ( z < N_PASSES*2 && x < pathSizes[z - N_PASSES]) {
      const char p = z - N_PASSES;

      sMinCost[p].backward = sPrevMinCost[p][0].backward;

      backwardCPt[p].x -= g_pathsDirections_d[p].x;
      backwardCPt[p].y -= g_pathsDirections_d[p].y;

      const float iI =
        iImage[backwardCPt[p].y*dsiDim.x + backwardCPt[p].x];

      sP2Adjust[p].backward = P2/abs(iI - sLastIntensity[p].backward);

      sLastIntensity[p].backward = iI;

      sDsiBaseOff[p].backward = dsiDim.z*dsiDim.y*backwardCPt[p].x
        + dsiDim.z*backwardCPt[p].y;
    }

    __syncthreads();

    for (char p=0; p<N_PASSES; p++) {
      if ( x >= pathSizes[p] )
        continue;

      fLr[p] = pathCost(costDSI, sDsiBaseOff[p].forward + z,
                        sPrevCost[p][dz].forward,
                        sPrevCost[p][dz - 1].forward,
                        sPrevCost[p][dz + 1].forward,
                        sMinCost[p].forward, sP2Adjust[p].forward,
                        aggregDSI);
#if 1
      bLr[p] = pathCost(costDSI, sDsiBaseOff[p].backward + z,
                        sPrevCost[p][dz].backward,
                        sPrevCost[p][dz - 1].backward,
                        sPrevCost[p][dz + 1].backward,
                        sMinCost[p].backward, sP2Adjust[p].backward,
                        aggregDSI);
#endif
    }

    __syncthreads();

    for (char p=0; p<N_PASSES; p++) {
      if ( x >= pathSizes[p] )
        continue;

      sPrevCost[p][dz].forward = sPrevMinCost[p][z].forward = fLr[p];
      sPrevCost[p][dz].backward = sPrevMinCost[p][z].backward = bLr[p];
    }

    __syncthreads();
  }
}

void RunSemiGlobalDev(const tdv::Dim &tdv_dsiDim, const float *dsi,
                      const tdv::SGPathsDesc *pathsDesc,
                      const float *lorigin, bool zeroAggregDSI,
                      float *aggregDSI, float *dispImg)
{
  DSIDim dsiDim(DSIDimCreate(tdv_dsiDim));

  tdv::CUerrExp cuerr;
  tdv::CudaConstraits constraits;

  tdv::WorkSize wsz = constraits.imageWorkSize(tdv_dsiDim);

  static const SGPoint pathsDirections_h[SG_MAX_PASSES] = {
    {1, 0},
    {0, 1},
    {1, 1},
    {-1, 1}
  };

  if ( zeroAggregDSI )
    zeroDSIVolume<<<wsz.blocks, wsz.threads>>>(dsiDim, aggregDSI);

  cuerr << cudaMemcpyToSymbol(g_pathsDirections_d, pathsDirections_h, sizeof(SGPoint)*SG_MAX_PASSES);

#if 1
  semiglobalAggregVolKernel<<<dsiDim.x, dsiDim.z>>>(dsiDim, pathsDesc,
                                                    dsi, lorigin,
                                                    aggregDSI);
#endif

#if 1
  wtaKernel<<<wsz.blocks, wsz.threads>>>(dsiDim, aggregDSI,
                                         dsiDim.x,
                                         dsiDim.x*dsiDim.y,
                                         dispImg);
#endif

  cuerr << cudaThreadSynchronize();
}
