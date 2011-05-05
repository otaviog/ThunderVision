#include "benchmark.hpp"
#include "crosscorrelationdev.hpp"

TDV_NAMESPACE_BEGIN

void DevCrossCorrelationRun(int maxDisparity,
                            Dim dsiDim, float *leftImg_d, float *rightImg_d,
                            float *dsiMem);

void CrossCorrelationDev::updateImpl(FloatImage leftImg, FloatImage rightImg,
                                     DSIMem dsi)
{
    CudaBenchmarker bm;
    bm.begin();

    float *leftImg_d = leftImg.devMem();
    float *rightImg_d = rightImg.devMem();

    DevCrossCorrelationRun(dsi.dim().depth(), dsi.dim(),
                           leftImg_d, rightImg_d, dsi.mem());

    bm.end();
}

TDV_NAMESPACE_END
