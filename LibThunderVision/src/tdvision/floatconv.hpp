#ifndef TDV_FLOATCONV_HPP
#define TDV_FLOATCONV_HPP

#include <cv.h>
#include "workunitutil.hpp"
#include "floatimage.hpp"

TDV_NAMESPACE_BEGIN

class FloatConv: public MonoWorkUnit<IplImage*, FloatImage>
{        
protected:
    FloatImage updateImpl(IplImage *img);
};

TDV_NAMESPACE_END

#endif /* TDV_FLOATCONV_HPP */
