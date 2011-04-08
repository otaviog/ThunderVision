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
      
      float lI = tex2D(texLeftImg, x + col, y + row), 
        rI = tex2D(texRightImg, x + disp + col, y + row);   
      
      sum += (lI - rI)*(lI - rI);
    }
  
  return sum;
}

__global__ void ssdKern(const int maxDisparity, const dim3 dim, float *dsiMem)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;     

  if ( x < dim.x && y < dim.y ) {    
    for (int disp=0; (disp < maxDisparity) && (x + disp) < dim.x; disp++) {   
      float ssdValue = ssdAtDisp(x, y, disp);

      const int volOffset = (dim.x*dim.y)*disp + y*dim.x + x;
      // volOffset =  disp + dim.z*x + (dim.z*dim.x)*y;
      dsiMem[volOffset] = ssdValue;
    }    
    
  }
}

TDV_NAMESPACE_BEGIN

void DevSSDRun(int maxDisparity,
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
    
  CudaConstraits constraits;  
  WorkSize ws = constraits.imageWorkSize(dsiDim);
  
  ssdKern<<<ws.blocks, ws.threads>>>(maxDisparity, 
                                     dim3(dsiDim.width(), dsiDim.height(),
                                          dsiDim.depth()),
                                     dsiMem); 
}

TDV_NAMESPACE_END