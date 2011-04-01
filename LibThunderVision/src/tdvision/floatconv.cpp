#include "floatconv.hpp"

TDV_NAMESPACE_BEGIN

FloatImage FloatConv::updateImpl(CvMat *img)
{
    return FloatImage(img);
}

TDV_NAMESPACE_END
