#ifndef TDV_CPYTOCPU_HPP
#define TDV_CPYTOCPU_HPP

#include <tdvbasic/common.hpp>
#include "pipe.hpp"
#include "workunitutil.hpp"
#include "floatimage.hpp"

TDV_NAMESPACE_BEGIN

class CpyImageToCPU: public MonoWorkUnit<FloatImage, FloatImage>
{
public:
    FloatImage updateImpl(FloatImage);    
};

TDV_NAMESPACE_END

#endif /* TDV_CPYTOCPU_HPP */
