#include "floatconv.hpp"

TDV_NAMESPACE_BEGIN

FloatImage FloatConv::updateImpl(IplImage *img)
{
    return FloatImage(img);
}

TDV_NAMESPACE_END
