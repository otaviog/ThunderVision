#include <cuda.h>
#include "cudaconstraits.hpp"
#include "dim.hpp"
#include "cuerr.hpp"
#include "dsimemutil.h"

__global__ void wtaKernel(const DSIDim dim, 
                          const float *dsi, 
                          const uint width,
                          const uint maxOffset, 
                          float *outimg)
{
  const uint x = blockDim.x*blockIdx.x + threadIdx.x;
  const uint y = blockDim.y*blockIdx.y + threadIdx.y;

  const uint offset = width*y + x;
  
  if ( offset < maxOffset ) {
    float leastDiff = CUDART_INF_F;
    uint wonDisparity = 0;
    
    for (uint d=0; d<dim.z && (d + x) < dim.x; d++) {      
      const float diff = dsiIntensity(dim, x, y, d, dsi);
            
      if ( diff < leastDiff ) {
        leastDiff = diff;
        wonDisparity = d;
      }
    }
    
    outimg[offset] = float(wonDisparity)/float(dim.z);            
    //outimg[offset] = 1.0f;
  }
}

void DevWTARun(float *dsi, const tdv::Dim &dsiDim, float *outimg)
{  
  DSIDim ddim(DSIDimCreate(dsiDim));

  tdv::CudaConstraits constraits;
  tdv::WorkSize wsz = constraits.imageWorkSize(dsiDim);
  
  wtaKernel<<<wsz.blocks, wsz.threads>>>(ddim, dsi, 
                                         dsiDim.width(),
                                         dsiDim.width()*dsiDim.height(),
                                         outimg); 
}