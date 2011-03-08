#include "dynamicprogdev.hpp"

__global__ void dynamicProgKernel(float *dsi, dim3 dsiDim, float *outimg)
{
    const uint x;
    const uint y;

    for ( int t=0; t<dsiDim.z; t++)
    {
        dsi[x,y] = min(dsi[x-1, y-1], dsi[x, y-1], dsi[x+1, y-1]);
        
    }
        
}

TDV_NAMESPACE_BEGIN

bool DynamicProgDev::update()
{
}

TDV_NAMESPACE_END
