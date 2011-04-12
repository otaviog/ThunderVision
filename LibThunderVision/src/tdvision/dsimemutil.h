#ifndef TDV_DSIUTIL_H
#define TDV_DSIUTIL_H

#define DSI_HIGHDSI_VALUE 999999.0

#include "cuerr.hpp"

inline __host__ __device__ uint dsiOffset(
    uint x, uint y, uint z)
{
    return z + g_dsiDim.z*x + g_dsiDim.x*g_dsiDim.z*y;
}

inline __host__ __device__ float dsiIntensityClamped(
    uint x, uint y, uint z, const float *dsi)
{    
    const uint offset = dsiOffset(x, y, z);   
    return offset < g_dsiDim.maxOffset ? dsi[offset] : DSI_HIGHDSI_VALUE; 
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

#endif /* TDV_DSIUTIL_H */
