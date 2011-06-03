#include "birchfieldcostdev.hpp"

TDV_NAMESPACE_BEGIN

Benchmark BirchfieldCostRun(Dim dsiDim, int maxDisparity,
                            float *leftImg, float *rightImg,
                            float *dsiMem);

void BirchfieldCostDev::updateImpl(FloatImage leftImg, FloatImage rightImg,
                                   DSIMem dsi)
{
    float *leftImg_d = leftImg.devMem();
    float *rightImg_d = rightImg.devMem();

    Benchmark mark = BirchfieldCostRun(dsi.dim(), dsi.dim().depth(),
                                       leftImg_d, rightImg_d, dsi.mem());
    
    m_benchmark.addProbe(mark);
}

TDV_NAMESPACE_END
