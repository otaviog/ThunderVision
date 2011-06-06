#include "semiglobaldev.hpp"

void RunSemiGlobalDev(const tdv::Dim &tdv_dsiDim, const float *dsi, 
                      const tdv::SGPathsDesc *pathsDesc,
                      const float *lorigin, bool zeroAggregDSI,
                      float *aggregDSI, float *dispImg);

TDV_NAMESPACE_BEGIN

void SemiGlobalDev::updateImpl(DSIMem dsi, FloatImage outimg)
{
    float *outimg_d = outimg.devMem();        
    float *aggregDSI = m_aggregDSI.mem(dsi.dim().size()*sizeof(float));
    
    SGPathsDesc *pdesc = m_sgPaths.getDesc(outimg.dim());
    
    RunSemiGlobalDev(dsi.dim(), dsi.mem(), 
                     pdesc, dsi.leftOrigin().devMem(), 
                     m_zeroAggregDSI, aggregDSI, outimg_d);        
    m_zeroAggregDSI = false;
}

void SemiGlobalDev::finished()
{
    m_aggregDSI.unalloc();
    m_sgPaths.unalloc();
    m_zeroAggregDSI = true;
}

TDV_NAMESPACE_END
