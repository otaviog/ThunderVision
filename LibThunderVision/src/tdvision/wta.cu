#include <cuda.h>
#include <math_constants.h>
#include "cudaconstraits.hpp"
#include "dim.hpp"
#include "cuerr.hpp"
#include "dsimemutil.h"

__global__ void wtaKernel(const dim3 dsiDim, 
                          cudaPitchedPtr dsiMem,
                          float *outimg)
{
  const uint x = blockDim.x*blockIdx.x + threadIdx.x;
  const uint y = blockDim.y*blockIdx.y + threadIdx.y;
  
  if ( x < dsiDim.x && y < dsiDim.y ) {
    float *dsiRow = (float*) (((char*) dsiMem.ptr) + 
                              DSI_GET_ROW_INCR(dsiMem, dsiDim, x, y));
    
    float leastDiff = CUDART_INF_F;
    uint wonDisparity = 0;            
    
    for (uint d=0; d<dsiDim.z && (d + x) < dsiDim.x; d++) {      
      const float diff = dsiRow[d]; 

      if ( diff < leastDiff ) {
        leastDiff = diff;
        wonDisparity = d;        
      }
      
      dsiRow[d] = 0.0f;
    }
    outimg[dsiDim.x*y + x] = float(wonDisparity)/float(dsiDim.z);
  }
}

#define MAX_DISP 512

__global__ void wtaKernel2(const dim3 dsiDim, 
                           cudaPitchedPtr dsiMem,
                           float *outimg)
{
  const size_t y = blockIdx.x;
  const size_t z = threadIdx.x;
  const size_t maxX = dsiDim.x;
  
  __shared__ float *sGblRow;
  __shared__ float sSharedRow[MAX_DISP];
  __shared__ int sDispRow[MAX_DISP];
  
  for (size_t x=0; x<maxX; x++) {
    if ( z == 0 ) {
      sGblRow = dsiGetRow(dsiMem, dsiDim.y, x, y);
    }  
    __syncthreads();
  
    sSharedRow[z] = sGblRow[z];
    sGblRow[z] = 0.0f;
    sDispRow[z] = z;

    __syncthreads();
    
    int i = dsiDim.z>>1;
    
    while ( i != 0 ) {
      if ( z < i ) {
        const float v1 = sSharedRow[z];
        const float v2 = sSharedRow[z + i];
        
        if ( v1 < v2 ) {
          sSharedRow[z] = v1;
        } else {
          sSharedRow[z] = v2;
          sDispRow[z] = sDispRow[z + i];
        }
      }
      __syncthreads();      
      i = i >> 1;
    }    
    
    __syncthreads();
    
    if ( z == 0 ) {
      outimg[dsiDim.x*y + x] = float(sDispRow[0])/float(dsiDim.z);
    }
  }
}

TDV_NAMESPACE_BEGIN

void WTARunDev(const tdv::Dim &dsiDim, cudaPitchedPtr dsiMem, float *outimg)
{  
  tdv::CudaConstraits constraits;
  tdv::WorkSize wsz = constraits.imageWorkSize(dsiDim);
  
#if 0
  wtaKernel<<<wsz.blocks, wsz.threads>>>(tdvDimTo(dsiDim), dsiMem, outimg);   
#else
  wtaKernel2<<<dsiDim.height(), dsiDim.depth()>>>(tdvDimTo(dsiDim), dsiMem, outimg);   
#endif
}

TDV_NAMESPACE_END