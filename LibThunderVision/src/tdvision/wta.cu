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
    float *dsiRow = dsiGetRow(dsiMem, dsiDim.x, x, y);
    
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

TDV_NAMESPACE_BEGIN

void WTARunDev(const tdv::Dim &dsiDim, cudaPitchedPtr dsiMem, float *outimg)
{  
  tdv::CudaConstraits constraits;
  tdv::WorkSize wsz = constraits.imageWorkSize(dsiDim);
  
  wtaKernel<<<wsz.blocks, wsz.threads>>>(tdvDimTo(dsiDim), dsiMem, outimg);   
}

TDV_NAMESPACE_END