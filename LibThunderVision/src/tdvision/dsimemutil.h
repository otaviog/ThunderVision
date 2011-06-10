#ifndef TDV_DSIUTIL_H
#define TDV_DSIUTIL_H

#include "cudamisc.hpp"

#define DSI_GET_ROW_INCR(pptr, dsiDim, px, py) pptr.pitch*dsiDim.y*px + pptr.pitch*py
#define DSI_GET_ROWF(pptr, dsiDim, px, py) (float*) (((char*) pptr.ptr) + DSI_GET_ROW_INCR(pptr, dsiDim, px, py))

inline __device__ float* dsiGetRow(cudaPitchedPtr &pptr, ushort height, 
                                   ushort x, ushort y)
{
  return (float*) ( ((char*) pptr.ptr) + pptr.pitch*height*x
                    + pptr.pitch*y);
}

inline __device__ const float* dsiGetRow(const cudaPitchedPtr &pptr, 
                                         ushort height, 
                                         ushort x, ushort y)
{
  return (float*) ( ((char*) pptr.ptr) + pptr.pitch*height*x
                    + pptr.pitch*y);
}

inline __device__ char* dsiGetRowB(cudaPitchedPtr pptr, ushort width, 
                                   ushort x, ushort y)
{
  return ( ((char*) pptr.ptr) + pptr.pitch*width*y
           + pptr.pitch*x);
}

inline __device__ float dsiGetValue(
    cudaPitchedPtr pptr, ushort w, ushort x, ushort y, ushort z)
{
    return dsiGetRow(pptr, w, x, y)[z];
}

inline __device__ char dsiGetValueB(
    cudaPitchedPtr pptr, ushort w, ushort x, ushort y, ushort z)
{
    return dsiGetRowB(pptr, w, x, y)[z];
}


    
#endif /* TDV_DSIUTIL_H */
