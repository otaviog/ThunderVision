#include <cuda.h>
#include "cudaconstraits.hpp"
#include "dim.hpp"

#define g_dsiDim g_wtaDsiDim

#define DSI_HIGHDSI_VALUE 999999.0

#include "cuerr.hpp"

struct DSIDim
{
    uint x, y, z;
    uint maxOffset;
};

__constant__ DSIDim g_dsiDim[1];

inline __host__ __device__ uint dsiOffset(
    uint x, uint y, uint z)
{
    return z + g_dsiDim[0].z*x + g_dsiDim[0].x*g_dsiDim[0].z*y;
}

inline __host__ __device__ float dsiIntensityClamped(
    uint x, uint y, uint z, const float *dsi)
{    
    const uint offset = dsiOffset(x, y, z);   
    return offset < g_dsiDim[0].maxOffset ? dsi[offset] : DSI_HIGHDSI_VALUE; 
}

inline __host__ __device__ float dsiIntensity(
    uint x, uint y, uint z, const float *dsi)
{    
    return dsi[dsiOffset(x, y, z)];
}

inline __host__ __device__ void dsiSetIntensity(
    uint x, uint y, uint z, float value, float *dsi)
{
    dsi[dsiOffset(x, y, z)] = value;
}

inline __host__ void dsiSetInfo(dim3 dsiDim, uint maxOffset)
{
  tdv::CUerrExp err;
  
  DSIDim dim;
  dim.x = dsiDim.x;
  dim.y = dsiDim.y;
  dim.z = dsiDim.z;
  dim.maxOffset = maxOffset;

  err << cudaMemcpyToSymbol(g_dsiDim, &dim, sizeof(DSIDim));
  
//  err << cudaMemcpyToSymbol(g_dsiDim, &dim, sizeof(DSIDim));
}

__global__ void wtaKernel(float *dsi, const uint maxOffset, 
                          float *outimg)
{
  const uint x = blockDim.x*blockIdx.x + threadIdx.x;
  const uint y = blockDim.y*blockIdx.y + threadIdx.y;

  const uint offset = blockDim.x*gridDim.x*y + x;
  
  if ( offset < maxOffset ) {
    float leastDiff = 1000.0f;
    uint wonDisparity = 0;
    
    for (uint d=0; d<g_dsiDim[0].z && (d + x) < g_dsiDim[0].x; d++) {      
      const float diff = dsiIntensity(x, y, d, dsi);
      
      if ( diff < leastDiff ) {
        leastDiff = diff;
        wonDisparity = d;
      }
    }
    
    //outimg[offset] = float(155)/float(g_dsiDim[0].z);    
    outimg[offset] = float(g_dsiDim[0].x);
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
  
  wtaKernel<<<wsz.blocks, wsz.threads>>>(dsi,
                                         dsiDim.width()*dsiDim.height(),
                                         outimg);
}