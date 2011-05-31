#include "semiglobaldev.hpp"

void RunSemiGlobalDev(const tdv::Dim &tdv_dsiDim, const float *dsi, 
                      const float *lorigin, float *aggregDSI, float *dispImg);

TDV_NAMESPACE_BEGIN

void SemiGlobalDev::updateImpl(DSIMem dsi, FloatImage outimg)
{
    float *outimg_d = outimg.devMem();
        
    RunSemiGlobalDev(dsi.dim(), dsi.mem(), dsi.leftOrigin().devMem(), 
                     m_aggregDSI.mem(dsi.dim().size()), outimg_d);
}

void SemiGlobalDev::finished()
{
    m_aggregDSI.unalloc();
}

TDV_NAMESPACE_END