#include <cuda.h>
#include "cudaconstraits.hpp"
#include "dim.hpp"

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
      const uint doffset = dsiDim.x*dsiDim.y*d + dsiDim.x*y + x;
      const float diff = dsi[doffset];
      
      if ( diff < leastDiff ) {
        leastDiff = diff;
        wonDisparity = d;
      }
    }
    //wonDisparity = 100;
    outimg[offset] = float(wonDisparity)/float(dsiDim.z);
  }
}

void DevWTARun(float *dsi, const tdv::Dim &dsiDim, float *outimg)
{
  
  tdv::CudaConstraits constraits;
  tdv::WorkSize wsz = constraits.imageWorkSize(dsiDim);
  
  wtaKernel<<<wsz.blocks, wsz.threads>>>(dsi, dim3(dsiDim.width(), 
                                                   dsiDim.height(), 
                                                   dsiDim.depth()), 
                                         dsiDim.width()*dsiDim.height(),
                                         outimg);
}