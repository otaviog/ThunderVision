#include "benchmark.hpp"
#include "crosscorrelationdev.hpp"

TDV_NAMESPACE_BEGIN

void CrossCorrelationDevRun(Dim dsiDim, float *leftImg_d, float *rightImg_d,
                            cudaPitchedPtr dsiMem);

void CrossCorrelationDev::updateImpl(FloatImage leftImg, FloatImage rightImg,
                                     DSIMem dsi)
{
    float *leftImg_d = leftImg.devMem();
    float *rightImg_d = rightImg.devMem();

    CrossCorrelationDevRun(dsi.dim(), leftImg_d, rightImg_d, 
                           dsi.mem());    
}

TDV_NAMESPACE_END
