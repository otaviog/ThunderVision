#include "benchmark.hpp"
#include "ssddev.hpp"

TDV_NAMESPACE_BEGIN

void DevSSDRun(Dim dsiDim, int maxDisparity,
               float *leftImg, float *rightImg,
               float *dsiMem);

void SSDDev::updateImpl(FloatImage leftImg, FloatImage rightImg,
                        DSIMem dsi)
{
    float *leftImg_d = leftImg.devMem();
    float *rightImg_d = rightImg.devMem();

    DevSSDRun(dsi.dim(), dsi.dim().depth(),
              leftImg_d, rightImg_d, dsi.mem());    
        
}

TDV_NAMESPACE_END
