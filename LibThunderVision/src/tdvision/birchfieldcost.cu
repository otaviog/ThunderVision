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
  
  for (int v=x; v < x + 8; v++) {    
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

__global__ void birchfieldKernTexture(const DSIDim dsiDim, 
                                      const int maxDisparity, 
                                      float *dsiMem)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;     

  if ( x < dsiDim.x && y < dsiDim.y ) {
    const uint baseOffset = dsiDim.z*dsiDim.y*x + dsiDim.z*y;
    
    for (int disp=0; (disp < maxDisparity); disp++) {   
      float value = CUDART_INF_F;
      
      if ( x - disp >= 0 ) {       
        value = costAtDisp(x, y, disp); 
      }
      
      dsiMem[baseOffset + disp] = value;      
    }    
    
  }
}

#define min3(a, b, c) min(a, min(b, c))
#define max3(a, b, c) max(a, max(b, c))

#define MAX_LINE_SIZE 1024
#define BT_N 4

__global__ void birchfieldKernSharedMem(const DSIDim dsiDim, 
                               const float *leftImg, const float *rightImg,
                               const int maxDisparity, float *dsiMem)
{
  const uint x = threadIdx.x;
  const uint y = blockIdx.x;
  
  __shared__ float leftScanLine[MAX_LINE_SIZE + 2];
  __shared__ float rightScanLine[MAX_LINE_SIZE + 2];
  
  const uint cPixelOffset = y*dsiDim.x + x;
  const uint dx = x + 1;
  
  leftScanLine[dx] = leftImg[cPixelOffset];
  rightScanLine[dx] = rightImg[cPixelOffset];
  
  if ( x == 0 ) {
    leftScanLine[0] = 0.0f;
    rightScanLine[0] = 0.0f;    
  }
  
  __syncthreads();
    
  const uint dsiBaseOffset = dsiDim.z*dsiDim.y*x + dsiDim.z*y;
  
  for (int disp=0; disp < maxDisparity; disp++) {   
    float costValue = CUDART_INF_F;    
    
    if ( static_cast<int>(x) - disp >= 0 ) {       
      costValue = 0.0f;      
      
      for (uint v=dx; v < dx + BT_N; v++) {  
        const uint vd = v - disp;
          
        const float lI = leftScanLine[dx];
        const float rI = rightScanLine[vd];  
      
        const float laI = 0.5f*(lI + leftScanLine[dx - 1]);
        const float lbI = 0.5f*(lI + leftScanLine[dx + 1]);
  
        const float raI = 0.5f*(rI + rightScanLine[vd - 1]);
        const float rbI = 0.5f*(rI + rightScanLine[vd + 1]);

        const float lImi = min3(laI, lbI, lI);
        const float lIma = max3(laI, lbI, lI);

        const float rImi = min3(raI, rbI, rI);
        const float rIma = max3(raI, rbI, rI);
    
        costValue += min(max3(0.0f, lI - rIma, rImi - lI),
                         max3(0.0f, rI - lIma, lImi - rI));              
      }            
    }
    
    dsiMem[dsiBaseOffset + disp] = costValue;
  }          
}

TDV_NAMESPACE_BEGIN

static void TextureBirchfieldRun(Dim dsiDim, int maxDisparity,
                                 float *leftImg_d, float *rightImg_d,
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
  birchfieldKernTexture<<<ws.blocks, ws.threads>>>(ddim, maxDisparity, 
                                                   dsiMem);  

}

static void SharedMemBirchfieldRun(Dim dsiDim, int maxDisparity,
                                   float *leftImg_d, float *rightImg_d,
                                   float *dsiMem)
{
  CUerrExp err;
  DSIDim ddim(DSIDimCreate(dsiDim));      
  
  birchfieldKernSharedMem<<<ddim.y, ddim.x>>>(ddim, leftImg_d, rightImg_d,
                                              maxDisparity, dsiMem); 
}

void BirchfieldCostRun(Dim dsiDim, int maxDisparity,
                       float *leftImg_d, float *rightImg_d,
                       float *dsiMem)
{
    cudaDeviceProp prop;    
    cudaGetDeviceProperties(&prop, 0);
    
    if ( dsiDim.width() <= prop.maxThreadsPerBlock ) {
      SharedMemBirchfieldRun(dsiDim, maxDisparity,
                             leftImg_d, rightImg_d,
                             dsiMem);      
    }
    else {
      TextureBirchfieldRun(dsiDim, maxDisparity,
                           leftImg_d, rightImg_d,
                           dsiMem);
    }

    cudaThreadSynchronize();
}

TDV_NAMESPACE_END
