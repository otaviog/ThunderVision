#include <climits>
#include "cuerr.hpp"
#include "wtacpu.hpp"
#include <iostream>

TDV_NAMESPACE_BEGIN

inline size_t dsiOffset(
    const Dim &dim, size_t x, size_t y, size_t z)
{
    return z + dim.depth()*y + dim.height()*dim.depth()*x;
}

void WTACPU::wta(const Dim &dsiDim, const float *dsi, float *dispImg)
{
    for (size_t y=0; y<dsiDim.height(); y++)
    {
        for (size_t x=0; x<dsiDim.width(); x++)
        {
            float minAggreg = std::numeric_limits<float>::infinity();
            int minDisp = 0;

            for (size_t d=0; d<dsiDim.depth(); d++)
            {
                const size_t dsiOff = dsiOffset(dsiDim, x, y, d);
                const float value = dsi[dsiOff];

                if ( value < minAggreg )
                {
                    minAggreg = value;
                    minDisp = d;
                }
            }

            dispImg[dsiDim.width()*y + x] =
                float(minDisp)/float(dsiDim.depth());
        }
    }
}

void WTACPU::updateImpl(DSIMem mem, FloatImage outimg)
{   
    const Dim &dim = mem.dim();
    
    boost::scoped_array<float> dsi((float*) mem.toCpuMem());    
    float *imgData_h = outimg.cpuMem()->data.fl;
    
    CudaBenchmarker bMarker;
    bMarker.begin();
    
    wta(dim, dsi.get(), imgData_h);    
    
    bMarker.end();                
    std::cout << bMarker.elapsedTime().millisecs() << std::endl;
}


TDV_NAMESPACE_END
