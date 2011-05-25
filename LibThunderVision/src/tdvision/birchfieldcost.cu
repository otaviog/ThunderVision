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
  float sum = 0.0f;
  
  for (int v=x - 12; v < x + 12; v++) {    
    const float lI = tex2D(texLeftImg, v, y);      
    const float rI = tex2D(texRightImg, v - disp, y);   
      
    const float laI = 0.5f*(lI + tex2D(texLeftImg, v - 1, y));
    const float lbI = 0.5f*(lI + tex2D(texLeftImg, v + 1, y));
  
    const float raI = 0.5f*(rI + tex2D(texRightImg, v - disp - 1, y));
    const float rbI = 0.5f*(rI + tex2D(texRightImg, v - disp + 1, y));

    const float lImi = min3(laI, lbI, lI);
    const float lIma = max3(laI, lbI, lI);

    const float rImi = min3(raI, rbI, rI);
    const float rIma = max3(raI, rbI, rI);
    
    sum += min(max3(0.0f, lI - rIma, rImi - lI),
               max3(0.0f, rI - lIma, lImi - rI));
  }
  
  return sum;
}

__global__ void birchfieldKern(const DSIDim dsiDim, const int maxDisparity, 
                               float *dsiMem)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;     

  if ( x < dsiDim.x && y < dsiDim.y ) {
    
    for (int disp=0; (disp < maxDisparity); disp++) {   
      float value = CUDART_INF_F;
      
      if ( x - disp >= 0 ) {       
        value = costAtDisp(x, y, disp); 
      }
      
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