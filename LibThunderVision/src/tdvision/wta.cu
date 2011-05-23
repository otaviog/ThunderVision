#include <cuda.h>
#include "cudaconstraits.hpp"
#include "dim.hpp"
#include "cuerr.hpp"
#include "dsimemutil.h"

__global__ void wtaKernel(const DSIDim dim, const float *dsi, 
                          const uint maxOffset, 
                          float *outimg)
{
  const uint x = blockDim.x*blockIdx.x + threadIdx.x;
  const uint y = blockDim.y*blockIdx.y + threadIdx.y;

  const uint offset = blockDim.x*gridDim.x*y + x;
  
  if ( offset < maxOffset ) {
    float leastDiff = 9999999.0f;
    uint wonDisparity = 0;
    
    for (uint d=0; d<dim.z && (d + x) < dim.x; d++) {      
      const float diff = dsiIntensity(dim, x, y, d, dsi);
            
      if ( diff < leastDiff ) {
        leastDiff = diff;
        wonDisparity = d;
      }
    }
    
    outimg[offset] = float(wonDisparity)/float(dim.z);            
  }
}

void DevWTARun(float *dsi, const tdv::Dim &dsiDim, float *outimg)
{  
  dim3 dsiDim_c(dsiDim.width(), 
                dsiDim.height(), 
                dsiDim.depth());
  DSIDim ddim(DSIDimCreate(dsiDim));

  tdv::CudaConstraits constraits;
  tdv::WorkSize wsz = constraits.imageWorkSize(dsiDim);
  
  wtaKernel<<<wsz.blocks, wsz.threads>>>(ddim, dsi,
                                         dsiDim.width()*dsiDim.height(),
                                         outimg);
}