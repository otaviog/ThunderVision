#include "dynamicprogdev.hpp"

TDV_NAMESPACE_BEGIN

void DynamicProgDevRun(const tdv::Dim &dsiDim, 
                       const cudaPitchedPtr costDSI,
                       cudaPitchedPtr pathDSI,
                       float *lastSumCosts,
                       float *dispImg);

void DynamicProgDev::updateImpl(DSIMem dsi, FloatImage outimg)
{
    float *outimg_d = outimg.devMem();    
    const Dim dsiDim = dsi.dim();
    
    DynamicProgDevRun(dsiDim, 
                      dsi.mem(),
                      m_pathDSI.mem(dsiDim),
                      (float*) m_lastCostsMem.mem(
                          dsiDim.depth()*dsiDim.height()*sizeof(float)),
                      outimg_d);        
}

void DynamicProgDev::finished()
{
    m_pathDSI.unalloc();
    m_lastCostsMem.unalloc();
}

TDV_NAMESPACE_END
