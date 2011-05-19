#include <cuda_runtime.h>
#include <cuda.h>
#include "cuerr.hpp"
#include "cudaconstraits.hpp"
#include "dsimemutil.h"

texture<float, 1> texLeftImg;
texture<float, 1> texRightImg;

static const int SAMPLES = 9;
static const int DIM = 3;

#define PI_CONST 3.14159265358979323846f

__device__ float gauss(float value)
{
  return exp(-PI_CONST*value*value);
}

__device__ int index(int seq, int xm, int ym, int width)
{
  return (ym - DIM/2 + seq/DIM)*width + (xm - DIM/2) + seq - (seq/DIM)*DIM;
}

__device__ float mutualInfoAtDisp(int x, int y, int disp, int width)
{
  float Bl, Br, Blr;
  Bl = Br = Blr = 0.0f;

  for (int i=0; i<SAMPLES; i++) {
    float Al, Ar, Alr;
    Al = Ar = Alr = 0.0f;
    
    const uint iOffset = index(i, x - disp, y, width);
    const uint iLocalOffset = index(i, x, y, width);
    
    for (int j=0; j<SAMPLES; j++) {      
      const uint jLocalOffset = index(j, x, y, width);
      const uint jOffset = index(j, x - disp, y, width);

      const float Gl = gauss(tex1Dfetch(texLeftImg, iLocalOffset) - tex1Dfetch(texLeftImg, jLocalOffset));
      const float Gr = gauss(tex1Dfetch(texRightImg, iOffset) - tex1Dfetch(texRightImg, jOffset));
      
      Al += Gl;
      Ar += Gr;
      Alr += Gl*Gr;
    }

    Bl += log(Al/SAMPLES);
    Br += log(Ar/SAMPLES);
    Blr += log(Alr/SAMPLES);
  }
  
  return -(Bl + Br - Blr)/SAMPLES;
  
}

__global__ void mutualInformation(const DSIDim dsiDim, const int maxDisparity, float *dsiMem)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if ( x < dsiDim.x && y < dsiDim.y ) {
    for (int disp=0; (disp < maxDisparity) && (x + disp) < dsiDim.x; disp++) {
      float miValue = mutualInfoAtDisp(x, y, disp, dsiDim.x);
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

  DSIDim ddim(DSIDimCreate(dsiDim));
  CudaConstraits constraits;
  WorkSize ws = constraits.imageWorkSize(dsiDim);
  mutualInformation<<<ws.blocks, ws.threads>>>(ddim, maxDisparity, dsiMem);
}

TDV_NAMESPACE_END