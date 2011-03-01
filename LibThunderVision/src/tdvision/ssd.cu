#include <cuda_runtime.h>
#include <cuda.h>
#include "cuerr.hpp"
#include "cudaconstraits.hpp"

texture<float, 2> texLeftImg;
texture<float, 2> texRightImg;

__device__ float ssdAtDisp(int x, int y, int disp)
{
  float sum = 0.0f;
  
  for (int row=-1; row<2; row++)
    for (int col=-1; col<2; col++) {
      float lI, rI; // I is for intensity      
      
      sum += (rI - lI)*(rI - lI);
    }
  
  return sum;
}

__global__ void ssdKern(const int maxDisparity, const dim3 dim, float *dsiMem)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;     

  if ( x < dim.x && y < dim.y ) {    
    for (int col=0; (col < maxDisparity) && (x + col) < dim.x; col++) {   
      float value = ssdAtDisp(x, y, col);

      const int volOffset = (dim.x*dim.y)*col + y*dim.x + x;
      dsiMem[volOffset] = value;
    }    
  }
}

TDV_NAMESPACE_BEGIN

void DevSSDRun(int maxDisparity,
               Dim imgDim, float *leftImg_d, float *rightImg_d,
               Dim dsiDim, float *dsiMem)
{
  CUerrExp err;
    
  err << cudaBindTexture2D(NULL, texLeftImg, leftImg_d, 
                           cudaCreateChannelDesc<float>(),
                           imgDim.width(), imgDim.height(),
                           imgDim.width()*sizeof(float));
  
  err << cudaBindTexture2D(NULL, texRightImg, rightImg_d, 
                           cudaCreateChannelDesc<float>(),
                           imgDim.width(), imgDim.height(),
                           imgDim.width()*sizeof(float));
  
  texLeftImg.addressMode[0] = texRightImg.addressMode[0] = cudaAddressModeWrap;
  texLeftImg.addressMode[1] = texRightImg.addressMode[1] = cudaAddressModeWrap;
  texLeftImg.normalized = texRightImg.normalized = false;
  texLeftImg.filterMode = texRightImg.filterMode = cudaFilterModePoint;
    
  CudaConstraits constraits;  
  WorkSize ws = constraits.imageWorkSize(imgDim);
  
  ssdKern<<<ws.blocks, ws.threads>>>(maxDisparity, 
                                     dim3(dsiDim.width(), dsiDim.height(),
                                          dsiDim.depth()),
                                     dsiMem); 
}

TDV_NAMESPACE_END