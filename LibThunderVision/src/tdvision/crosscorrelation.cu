#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda.h>
#include "cuerr.hpp"
#include "cudaconstraits.hpp"
#include "dsimemutil.h"

texture<float, 2> texLeftImg;
texture<float, 2> texRightImg;

__device__ float ccAtDisp(int x, int y, int disp)
{
  float domSum = 0.0f,
    lSum = 0.0f,
    rSum = 0.0f;

  for (int row=-1; row<2; row++)
    for (int col=-1; col<2; col++) {
      const float lValue = tex2D(texLeftImg, x + col, y + row);
      const float rValue = tex2D(texRightImg, x + col - disp, y + row);
      domSum += lValue*rValue;
       
      lSum += lValue*lValue;
      rSum += rValue*rValue;      
    }

  return abs(1.0f - domSum/sqrt(lSum*rSum));
}

__global__ void ccorrelationKern(const dim3 dsiDim, cudaPitchedPtr dsiMem)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if ( x < dsiDim.x && y < dsiDim.y ) {
    float *dsiRow = dsiGetRow(dsiMem, dsiDim.x, x, y);
    
    for (int disp=0; disp < dsiDim.z; disp++) {
      float ccValue = CUDART_INF_F;
      if ( x - disp >= 0)
        ccValue = ccAtDisp(x, y, disp);
      dsiRow[disp] = ccValue;
    }
    
  }
}

TDV_NAMESPACE_BEGIN

void CrossCorrelationDevRun(Dim dsiDim, float *leftImg_d, float *rightImg_d,
                            cudaPitchedPtr dsiMem)
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
  ccorrelationKern<<<ws.blocks, ws.threads>>>(tdvDimTo(dsiDim), dsiMem);   
}

TDV_NAMESPACE_END