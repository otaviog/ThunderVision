#include <cuda.h>
#include "cudaconstraits.hpp"
#include "dim.hpp"
#include "dsimemutil.h"

__global__ void wtaKernel(float *dsi, dim3 dsiDim, 
                          const uint maxOffset, float *outimg)
{
  const uint x = blockDim.x*blockIdx.x + threadIdx.x;
  const uint y = blockDim.y*blockIdx.y + threadIdx.y;

  const uint offset = blockDim.x*gridDim.x*y + x;
  
  if ( offset < maxOffset ) {
    float leastDiff = 1000.0f;
    uint wonDisparity = 0;
    
    for (uint d=0; d<dsiDim.z && (d + x) < dsiDim.x; d++) {      
      const float diff = dsiIntensity(x, y, d, dsi);
      
      if ( diff < leastDiff ) {
        leastDiff = diff;
        wonDisparity = d;
      }
    }

    outimg[offset] = float(wonDisparity)/float(dsiDim.z);
  }
}

void DevWTARun(float *dsi, const tdv::Dim &dsiDim, float *outimg)
{  
  tdv::CudaConstraits constraits;
  tdv::WorkSize wsz = constraits.imageWorkSize(dsiDim);
  
  dim3 dsiDim_c(dsiDim.width(), 
                dsiDim.height(), 
                dsiDim.depth());
  dsiSetInfo(dsiDim_c, dsiDim.size());
  wtaKernel<<<wsz.blocks, wsz.threads>>>(dsi, dsiDim_c, 
                                         dsiDim.width()*dsiDim.height(),
                                         outimg);
}