#include <cuda_runtime.h>
#include <cuda.h>
#include "cuerr.hpp"
#include "cudaconstraits.hpp"
#include "dsimemutil.h"

texture<float, 1> texLeftImg;
texture<float, 1> texRightImg;

static const int SAMPLES = 9;
static const int DIM = 3;

__device__ int index(int seq, int xm, int ym, int width)
{
  return (ym - DIM/2 + seq/DIM)*width + (xm - DIM/2) + seq - (seq/DIM)*DIM;
}

__global__ void zeroHistograms(const DSIDim histDim, int *lhistMem, int *rhistMem)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;
  
  for (int i=0; i<SAMPLES; i++) {
    lhistMem[dsiOffset(histDim, x, y, i)] = 0;
    rhistMem[dsiOffset(histDim, x, y, i)] = 0;
  }
}

__global__ void windowHistogram(const DSIDim histDim,
                                int *lhistMem, 
                                int *rhistMem)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;
    
  for (int i=0; i<SAMPLES; i++) {
    const int texIdx = index(i, x, y, histDim.x);
    const float lvalue = tex1Dfetch(texLeftImg, texIdx);
    const float rvalue = tex1Dfetch(texRightImg, texIdx);    
    
    lhistMem[dsiOffset(histDim, x, y, int(lvalue*(SAMPLES - 1)))] += 1;
    rhistMem[dsiOffset(histDim, x, y, int(rvalue*(SAMPLES - 1)))] += 1;
  }
  
}

__device__ float mutualInfoAtDisp(int x, int y, int disp, int width, 
                                  const DSIDim histDim,
                                  const int *lhistMem,
                                  const int *rhistMem)
{
  float mi = 0.0f;
  
  for (int i=0; i<SAMPLES; i++) {    
    const uint iOffset = index(i, x, y, width);    
    const float iValue = tex1Dfetch(texLeftImg, iOffset);
    const int histI = lhistMem[dsiOffset(histDim, x, y, int(iValue*(SAMPLES - 1)))];        
    
    for (int j=0; j<SAMPLES; j++) {      
      const uint jOffset = index(j, x - disp, y, width);      
      const float jValue = tex1Dfetch(texRightImg, jOffset);
      const int histJ = rhistMem[dsiOffset(histDim, x, y, int(jValue*(SAMPLES - 1)))];
      
      float probI = histI;      
      float probJ = histJ;                  
      float probIJ = 0;
      
      if ( histI < 1 )
        probIJ = 0;
      else if ( histI < 4 )
        probIJ = 1;
      else if ( histI < 6 )
        probIJ = 2;
      else if ( histI < 8 )
        probIJ = 3;
      else
        probIJ = 4;
      mi += probIJ*log(probIJ/(probI*probJ));      
    }
  }
  
  return 1.0 - mi;  
}

__global__ void mutualInformation(const DSIDim dsiDim, const int maxDisparity, 
                                  const DSIDim histDim, 
                                  const int *lhistMem, const int *rhistMem,
                                  float *dsiMem)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  if ( x < dsiDim.x && y < dsiDim.y ) {
    for (int disp=0; (disp < maxDisparity) && (x + disp) < dsiDim.x; disp++) {
      float miValue = mutualInfoAtDisp(x, y, disp, dsiDim.x, 
                                       histDim, lhistMem, rhistMem);
      dsiSetIntensity(dsiDim, x, y, disp, miValue, dsiMem);
    }
  }
}

TDV_NAMESPACE_BEGIN

void DevMutualInformationRun(int maxDisparity,
                             Dim dsiDim, float *leftImg_d, float *rightImg_d,
                             float *dsiMem)
{
  CUerrExp err;

  size_t offset;
  err << cudaBindTexture(&offset, texLeftImg, leftImg_d,
                         dsiDim.size()*sizeof(float));

  err << cudaBindTexture(NULL, texRightImg, rightImg_d,
                         dsiDim.size()*sizeof(float));

  texLeftImg.addressMode[0] = texRightImg.addressMode[0] = cudaAddressModeWrap;
  texLeftImg.normalized = texRightImg.normalized = false;
  texLeftImg.filterMode = texRightImg.filterMode = cudaFilterModePoint;
    
  CudaConstraits constraits;
  WorkSize ws = constraits.imageWorkSize(dsiDim);  
  
  DSIDim histDim(DSIDimCreate(Dim(dsiDim.width(), dsiDim.height(), SAMPLES)));;
  
  int *lhistMem, *rhistMem;  
  err << cudaMalloc(&lhistMem, histDim.maxOffset*sizeof(int));
  err << cudaMalloc(&rhistMem, histDim.maxOffset*sizeof(int));
  
  zeroHistograms<<<ws.blocks, ws.threads>>>(histDim, lhistMem, rhistMem);
  
  err << cudaThreadSynchronize();
  
  windowHistogram<<<ws.blocks, ws.threads>>>(histDim, lhistMem, rhistMem);
  
  err << cudaThreadSynchronize();
 
  DSIDim dDim(DSIDimCreate(dsiDim));
  mutualInformation<<<ws.blocks, ws.threads>>>(dDim, maxDisparity, 
                                               histDim, lhistMem, rhistMem,
                                               dsiMem);
  
  err << cudaFree(lhistMem);
  err << cudaFree(rhistMem);
  
}

TDV_NAMESPACE_END