#ifndef TDV_DSIUTIL_H
#define TDV_DSIUTIL_H

#include "cudamisc.hpp"

#define DSI_GET_ROW_INCR_H(pptr, height, px, py) pptr.pitch*height*px + pptr.pitch*py

#define DSI_GET_ROW_INCR(pptr, dsiDim, px, py) DSI_GET_ROW_INCR_H(pptr, dsiDim.y, px, py)

#define DSI_GET_ROWB(pptr, dsiDim, px, py) ((((char*) pptr.ptr) + DSI_GET_ROW_INCR(pptr, dsiDim, px, py)))

#define DSI_GET_ROWF(pptr, dsiDim, px, py) ((float*) DSI_GET_ROWB(pptr, dsiDim, px, py))


inline __device__ char* dsiGetRowB(cudaPitchedPtr pptr, ushort height, 
                                   ushort x, ushort y)
{
    return ( ((char*) pptr.ptr) + DSI_GET_ROW_INCR_H(pptr, height, x, y));

}

inline __device__ float* dsiGetRow(cudaPitchedPtr pptr, ushort height, 
                                   ushort x, ushort y)
{
    return (float*) dsiGetRowB(pptr, height, x, y);
}

template<typename TypePtr> 
inline __device__ TypePtr dsiGetRowT(cudaPitchedPtr pptr, dim3 dsiDim, 
                                   int x, int y)
{
    return (TypePtr) dsiGetRowB(pptr, dsiDim.y, x, y);
}

inline __device__ float dsiGetValue(
    cudaPitchedPtr pptr, ushort h, ushort x, ushort y, ushort z)
{
    return dsiGetRow(pptr, h, x, y)[z];
}

inline __device__ char dsiGetValueB(
    const cudaPitchedPtr pptr, ushort h, ushort x, ushort y, ushort z)
{
    return dsiGetRowB(pptr, h, x, y)[z];
}


    
#endif /* TDV_DSIUTIL_H */
