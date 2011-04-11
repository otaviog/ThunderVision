#ifndef TDV_DSIUTIL_H
#define TDV_DSIUTIL_H

#define DSI_HIGHDSI_VALUE 999999.0

struct DSIDim
{
    uint x, y, z;
};

__constant__ DSIDim g_dsiDim;
__constant__ uint g_dsiMaxOffset;

inline __host__ __device__ uint dsiOffset(
    uint x, uint y, uint z)
{
    return z + g_dsiDim.z*x + g_dsiDim.x*g_dsiDim.y*y;
}

inline __host__ __device__ float dsiIntensityClamped(
    uint x, uint y, uint z, const float *dsi)
{    
    const uint offset = dsiOffset(x, y, z);   
    return offset < g_dsiMaxOffset ? dsi[offset] : DSI_HIGHDSI_VALUE; 
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
  cudaMemcpyToSymbol(&g_dsiDim, &dsiDim, sizeof(dim3));
  cudaMemcpyToSymbol(&g_dsiMaxOffset, &maxOffset, sizeof(uint));    
}

#endif /* TDV_DSIUTIL_H */
