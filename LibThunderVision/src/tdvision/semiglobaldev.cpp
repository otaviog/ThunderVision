#include "semiglobaldev.hpp"

void RunSemiGlobalDev(const tdv::Dim &tdv_dsiDim, float *dsi, float *dispImg);

TDV_NAMESPACE_BEGIN

void SemiGlobalDev::updateImpl(DSIMem dsi, FloatImage outimg)
{
    float *outimg_d = outimg.devMem();    
    RunSemiGlobalDev(dsi.dim(), dsi.mem(), outimg_d);
}

TDV_NAMESPACE_END
