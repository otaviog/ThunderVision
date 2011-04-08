#ifndef TDV_DSIUTIL_H
#define TDV_DSIUTIL_H

__global__ __device__ dsiVolumeIndex(comst dim3 &dsiDim, uint x, uint y, uint z)
{
    return z + dim.z*x + dim.x*dim.y*y;
}

#endif /* TDV_DSIUTIL_H */
