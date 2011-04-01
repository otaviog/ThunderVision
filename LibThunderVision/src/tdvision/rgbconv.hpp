#ifndef TDV_RGBCONV_HPP
#define TDV_RGBCONV_HPP

#include <tdvbasic/common.hpp>
#include <cv.h>
#include "workunitutil.hpp"
#include "pipe.hpp"
#include "floatimage.hpp"

TDV_NAMESPACE_BEGIN

class RGBConv: public MonoWorkUnit<FloatImage, CvMat*>
{
public:
    RGBConv();

protected:
    CvMat* updateImpl(FloatImage image);
};

TDV_NAMESPACE_END

#endif /* TDV_RGBCONV_HPP */
