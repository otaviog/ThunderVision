#include "benchmark.hpp"
#include "crosscorrelationdev.hpp"

TDV_NAMESPACE_BEGIN

void DevCrossCorrelationRun(Dim dsiDim, int maxDisparity,
                            float *leftImg_d, float *rightImg_d,
                            float *dsiMem);

void CrossCorrelationDev::updateImpl(FloatImage leftImg, FloatImage rightImg,
                                     DSIMem dsi)
{
    float *leftImg_d = leftImg.devMem();
    float *rightImg_d = rightImg.devMem();

    DevCrossCorrelationRun(dsi.dim(), dsi.dim().depth(),
                                  leftImg_d, rightImg_d, dsi.mem());    
}

TDV_NAMESPACE_END
