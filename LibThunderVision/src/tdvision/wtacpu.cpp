#include <climits>
#include "cuerr.hpp"
#include "wtacpu.hpp"

TDV_NAMESPACE_BEGIN

inline size_t dsiOffset(
    const Dim &dim, size_t x, size_t y, size_t z)
{
    return z + dim.depth()*y + dim.height()*dim.depth()*x;
}

void WTACPU::updateImpl(DSIMem mem, FloatImage outimg)
{   
    CUerrExp cuerr;         
    const Dim &dim = mem.dim();
    
    boost::scoped_array<float> dsi(new float[dim.size()]);
    
    cuerr << cudaMemcpy(dsi.get(), mem.mem(), sizeof(float)*dim.size(),
                        cudaMemcpyDeviceToHost);

    float *imgData_h = outimg.cpuMem()->data.fl;
    for (size_t row=0; row<dim.height(); row++)
    {
        for (size_t col=0; col<dim.width(); col++)
        {
            float minCost = dsi.get()[dsiOffset(dim, col, row, 0)];
            size_t minDisp = 0;
            
            for (size_t d=1; d<dim.depth(); d++)
            {
                const float cost = dsi.get()[dsiOffset(dim, col, row, d)];
                if ( cost < minCost )
                {
                    minCost = cost;
                    minDisp = d;
                }
            }
            
            imgData_h[row*dim.width() + col] = float(minDisp)/float(dim.depth());            
        }
    }
}

TDV_NAMESPACE_END
