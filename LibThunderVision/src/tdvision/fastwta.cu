#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda.h>
#include "cuerr.hpp"
#include "cudaconstraits.hpp"
#include "dsimemutil.h"

texture<float, 2> texLeftImg;
texture<float, 2> texRightImg;

#define SSD_WIND_DIM 5
#define SSD_WIND_START -3
#define SSD_WIND_END 4

__device__ float ssdAtDisp2(int x, int y, int disp)
{
  float sum = 0.0f;

  for (int row=SSD_WIND_START; row<SSD_WIND_END; row++)
    for (int col=SSD_WIND_START; col<SSD_WIND_END; col++) {

      float lI = tex2D(texLeftImg, x + col, y + row),
        rI = tex2D(texRightImg, x + col - disp, y + row);

      sum += (lI - rI)*(lI - rI);
    }

  return sum;
}


__global__ void fastWtaKern(const dim3 dsiDim, float *dispImage)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y; 
  
  if ( x < dsiDim.x && y < dsiDim.y ) {
    float minSSD = CUDART_INF_F;
    int minDisp = 0;
    
    const int nDisps = min(dsiDim.z, x + 1);    
    for (int disp=0; disp < nDisps; disp++) {           
      const float ssdValue = ssdAtDisp2(x, y, disp);
      
      if ( ssdValue < minSSD ) {
        minSSD = ssdValue;
        minDisp = disp;
      }            
    }
    
    dispImage[dsiDim.x*y + x] = float(minDisp)/float(dsiDim.z);
  }
}

TDV_NAMESPACE_BEGIN

void FastWTADevRun(Dim dsiDim, float *leftImg_d, float *rightImg_d, 
                   float *dispImg)
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

  CudaConstraits constraits;
  WorkSize ws = constraits.imageWorkSize(dsiDim);
  
  fastWtaKern<<<ws.blocks, ws.threads>>>(tdvDimTo(dsiDim), dispImg);
  
  err << cudaThreadSynchronize();
}

TDV_NAMESPACE_END