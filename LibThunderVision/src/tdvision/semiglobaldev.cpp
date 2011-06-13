#include "semiglobaldev.hpp"

TDV_NAMESPACE_BEGIN

void SemiGlobalDevRun(const tdv::Dim &dsiDim, cudaPitchedPtr dsi,
                      const tdv::SGPath *pathsArray, size_t pathCount,
                      const float *lorigin, cudaPitchedPtr aggregDSI, 
                      float *dispImg, bool zeroAggregDSI);

void SemiGlobalDev::updateImpl(DSIMem dsi, FloatImage outimg)
{
    float *outimg_d = outimg.devMem();
    cudaPitchedPtr aggregDSI = m_aggregDSI.mem(dsi.dim());
    SGPath *paths = m_sgPaths.getDescDev(outimg.dim());

    SemiGlobalDevRun(dsi.dim(), dsi.mem(),
                     paths, m_sgPaths.pathCount(),
                     dsi.leftOrigin().devMem(), aggregDSI,
                     outimg_d, m_zeroAggregDSI);

    m_zeroAggregDSI = false;
}

void SemiGlobalDev::finished()
{
    m_aggregDSI.unalloc();
    m_sgPaths.unalloc();
    m_zeroAggregDSI = true;
}

TDV_NAMESPACE_END
