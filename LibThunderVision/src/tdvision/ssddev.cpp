#include "benchmark.hpp"
#include "ssddev.hpp"

TDV_NAMESPACE_BEGIN

void SSDDevRun(Dim dsiDim, float *leftImg_d, float *rightImg_d,
               cudaPitchedPtr dsiMem);

void SSDDev::updateImpl(FloatImage leftImg, FloatImage rightImg,
                        DSIMem dsi)
{
    float *leftImg_d = leftImg.devMem();
    float *rightImg_d = rightImg.devMem();

    SSDDevRun(dsi.dim(), leftImg_d, rightImg_d, dsi.mem());    
        
}

TDV_NAMESPACE_END
