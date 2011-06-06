#include <cuda.h>
#include "cudaconstraits.hpp"
#include "dim.hpp"
#include "cuerr.hpp"
#include "dsimemutil.h"

__global__ void wtaKernel(const DSIDim dim, 
                          float *dsi, 
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
    
    const uint dsiOffsetBase = dim.z*dim.y*x + dim.z*y;
    
    for (uint d=0; d<dim.z && (d + x) < dim.x; d++) {      
      const uint dsiOff = dsiOffsetBase + d;
      const float diff = dsi[dsiOff]; 

      if ( diff < leastDiff ) {
        leastDiff = diff;
        wonDisparity = d;        
      }
      
      dsi[dsiOff] = 0.0f;
    }
    
    outimg[offset] = float(wonDisparity)/float(dim.z);
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