#include <cuda_runtime.h>
#include <cuda.h>
#include "cuerr.hpp"
#include "cudaconstraits.hpp"
#include "dsimemutil.h"

texture<float, 2> texLeftImg;
texture<float, 2> texRightImg;

#define SSD_WIND_DIM 5
#define SSD_WIND_START -3
#define SSD_WIND_END 4

#define min3(a, b, c) min(a, min(b, c))
#define max3(a, b, c) max(a, max(b, c))

__device__ float costAtDisp(int x, int y, int disp)
{
    
  const float lI = tex2D(texLeftImg, x, y);      
  const float rI = tex2D(texRightImg, x - disp, y);   
      
  const float raI = 0.5f*(rI + tex2D(texRightImg, x + disp - 1, y));
  const float rbI = 0.5f*(rI + tex2D(texRightImg, x + disp + 1, y));
  
  const float rImi = min3(raI, rbI, rI);
  const float rIma = max3(raI, rbI, rI);
  
  return max3(0.0f, lI - rIma, rImi - lI);
}

__global__ void birchfieldKern(const DSIDim dsiDim, const int maxDisparity, 
                               float *dsiMem)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;     

  if ( x < dsiDim.x && y < dsiDim.y ) {
    
    for (int disp=0; (disp < maxDisparity) && (x - disp) >= 0; disp++) {   
      float value = costAtDisp(x, y, disp);      
      dsiSetIntensity(dsiDim, x, y, disp, value, dsiMem);
    }    
    
  }
}

TDV_NAMESPACE_BEGIN

void BirchfieldCostRun(int maxDisparity,
                       Dim dsiDim, float *leftImg_d, float *rightImg_d,
                       float *dsiMem)
{
  CUerrExp err;
    
  err << cudaBindTexture2D(NULL, texLeftImg, leftImg_d, 
                           cudaCreateChannelDesc<float>(),
                           dsiDim.width(), dsiDim.height(),
                           dsiDim.width()*sizeof(float));
  
  err << cudaBindTexture2D(NULL, texRightImg, rightImg_d, 
                           cudaCreateChannelDesc<float>(),
                           dsiDim.width(), dsiDim.height(),
                           dsiDim.width()*sizeof(float));
  
  texLeftImg.addressMode[0] = texRightImg.addressMode[0] = cudaAddressModeWrap;
  texLeftImg.addressMode[1] = texRightImg.addressMode[1] = cudaAddressModeWrap;
  texLeftImg.normalized = texRightImg.normalized = false;
  texLeftImg.filterMode = texRightImg.filterMode = cudaFilterModePoint;
    
  DSIDim ddim(DSIDimCreate(dsiDim));  
  
  CudaConstraits constraits;  
  WorkSize ws = constraits.imageWorkSize(dsiDim);  
  birchfieldKern<<<ws.blocks, ws.threads>>>(ddim, maxDisparity, dsiMem); 
}

TDV_NAMESPACE_END