#include "dynamicprogdev.hpp"

tdv::Benchmark RunDynamicProgDev(const tdv::Dim &tdv_dsiDim, float *dsi, float *dispImg);

TDV_NAMESPACE_BEGIN

void DynamicProgDev::updateImpl(DSIMem dsi, FloatImage outimg)
{
    float *outimg_d = outimg.devMem();    
    Benchmark marker = RunDynamicProgDev(dsi.dim(), dsi.mem(), outimg_d);
    
    m_marker.addProbe(marker);
}

TDV_NAMESPACE_END
