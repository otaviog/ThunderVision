#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda.h>
#include "cuerr.hpp"
#include "cudaconstraits.hpp"
#include "dsimemutil.h"

texture<float, 2> texLeftImg;
texture<float, 2> texRightImg;

__device__ float ccAtDisp(short x, short y, short disp)
{
  float domSum = 0.0f,
    lSum = 0.0f,
    rSum = 0.0f;

  for (short row=-1; row<2; row++)
    for (short col=-1; col<2; col++) {
      const float lValue = tex2D(texLeftImg, x + col, y + row);
      const float rValue = tex2D(texRightImg, x + col - disp, y + row);
      domSum += lValue*rValue;
       
      lSum += lValue*lValue;
      rSum += rValue*rValue;      
    }

  return 1.0f - domSum/sqrt(lSum*rSum);
}

__global__ void ccorrelationKern(const dim3 dsiDim, cudaPitchedPtr dsiMem)
{
  short x = blockIdx.x*blockDim.x + threadIdx.x;
  short y = blockIdx.y*blockDim.y + threadIdx.y;

  if ( x < dsiDim.x && y < dsiDim.y ) {
    float *dsiRow = dsiGetRow(dsiMem, dsiDim.y, x, y);
    
    const short  nDisps = min(dsiDim.z, x + 1);
    
    for (short disp=0; disp < nDisps; disp++) {
      dsiRow[disp] = ccAtDisp(x, y, disp);
    }
    
    for (short disp=nDisps; disp<dsiDim.z; disp++) {
      dsiRow[disp] = CUDART_INF_F;
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