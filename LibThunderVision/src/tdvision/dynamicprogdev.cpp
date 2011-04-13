#include "dynamicprogdev.hpp"

void RunDynamicProgDev(const tdv::Dim &tdv_dsiDim, float *dsi, float *dispImg);

TDV_NAMESPACE_BEGIN

void DynamicProgDev::updateImpl(DSIMem dsi, FloatImage outimg)
{
    float *outimg_d = outimg.devMem();    
    RunDynamicProgDev(dsi.dim(), dsi.mem(), outimg_d);
}

TDV_NAMESPACE_END
