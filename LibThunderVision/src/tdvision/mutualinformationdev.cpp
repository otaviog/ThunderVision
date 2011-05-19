#include "benchmark.hpp"
#include "mutualinformationdev.hpp"

TDV_NAMESPACE_BEGIN

void DevMutualInformationRun(
    int maxDisparity,
    Dim dsiDim, float *leftImg, float *rightImg,
    float *dsiMem);

void MutualInformationDev::updateImpl(
    FloatImage leftImg, FloatImage rightImg,
    DSIMem dsi)
{
    CudaBenchmarker bm;
    bm.begin();

    float *leftImg_d = leftImg.devMem();
    float *rightImg_d = rightImg.devMem();

    DevMutualInformationRun(dsi.dim().depth(), dsi.dim(),
                            leftImg_d, rightImg_d, dsi.mem());
    bm.end();
}

TDV_NAMESPACE_END
