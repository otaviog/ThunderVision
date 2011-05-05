#include "benchmark.hpp"
#include "ssddev.hpp"

TDV_NAMESPACE_BEGIN

void DevSSDRun(int maxDisparity,
               Dim dsiDim, float *leftImg, float *rightImg,
               float *dsiMem);

void SSDDev::updateImpl(FloatImage leftImg, FloatImage rightImg,
                        DSIMem dsi)
{
    CudaBenchmarker bm;
    bm.begin();

    float *leftImg_d = leftImg.devMem();
    float *rightImg_d = rightImg.devMem();

    DevSSDRun(dsi.dim().depth(), dsi.dim(),
              leftImg_d, rightImg_d, dsi.mem());
    bm.end();
}

TDV_NAMESPACE_END
