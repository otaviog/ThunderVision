#include "birchfieldcostdev.hpp"

TDV_NAMESPACE_BEGIN

void BirchfieldCostRun(Dim dsiDim,
                       float *leftImg_d, float *rightImg_d,
                       cudaPitchedPtr costDSI);

void BirchfieldCostDev::updateImpl(FloatImage leftImg, FloatImage rightImg,
                                   DSIMem dsi)
{
    float *leftImg_d = leftImg.devMem();
    float *rightImg_d = rightImg.devMem();

    BirchfieldCostRun(dsi.dim(), leftImg_d, rightImg_d, 
                      dsi.mem());    
}

TDV_NAMESPACE_END
