#ifndef TDV_DSIUTIL_H
#define TDV_DSIUTIL_H

#include "dim.hpp"
#include "cuerr.hpp"

struct DSIDim
{
    uint x, y, z;
    uint maxOffset;
};

#define DSI_HIGHDSI_VALUE 999999.0

inline __host__ __device__ uint dsiOffset(
    const DSIDim &dim, uint x, uint y, uint z)
{
    return z + dim.z*y + dim.y*dim.z*x;
}

inline __host__ __device__ float dsiIntensityClamped(
    const DSIDim &dim, uint x, uint y, uint z, const float *dsi)
{    
    const uint offset = dsiOffset(dim, x, y, z);   
    return offset < dim.maxOffset ? dsi[offset] : DSI_HIGHDSI_VALUE; 
}

inline __host__ __device__ float dsiIntensity(
    const DSIDim &dim, uint x, uint y, uint z, const float *dsi)
{    
    return dsi[dsiOffset(dim, x, y, z)];
}

inline __host__ __device__ void dsiSetIntensity(
    const DSIDim &dim, uint x, uint y, uint z, float value, 
    float *dsi)
{
    dsi[dsiOffset(dim, x, y, z)] = value;
}

inline DSIDim DSIDimCreate(const tdv::Dim &dim)
{
  tdv::CUerrExp err;
  
  DSIDim ddim;
  ddim.x = dim.width();
  ddim.y = dim.height();
  ddim.z = dim.depth();
  ddim.maxOffset = dim.size();
  
  return ddim;
}

#endif /* TDV_DSIUTIL_H */
