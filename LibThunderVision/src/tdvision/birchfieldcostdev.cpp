#include "birchfieldcostdev.hpp"

TDV_NAMESPACE_BEGIN

void BirchfieldCostRun(int maxDisparity,
                       Dim dsiDim, float *leftImg, float *rightImg,
                       float *dsiMem);

void BirchfieldCostDev::updateImpl(FloatImage leftImg, FloatImage rightImg,
                                   DSIMem dsi)
{
    float *leftImg_d = leftImg.devMem();
    float *rightImg_d = rightImg.devMem();

    BirchfieldCostRun(dsi.dim().depth(), dsi.dim(),
                      leftImg_d, rightImg_d, dsi.mem());

}

TDV_NAMESPACE_END
