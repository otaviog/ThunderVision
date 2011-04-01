#ifndef TDV_MEDIANFILTERDEV_HPP
#define TDV_MEDIANFILTERDEV_HPP

#include <tdvbasic/common.hpp>
#include "workunitutil.hpp"
#include "pipe.hpp"
#include "floatimage.hpp"

TDV_NAMESPACE_BEGIN

class MedianFilterDev: public MonoWorkUnit<FloatImage, FloatImage>
{
public:        
    MedianFilterDev()       
    { 
        workName("Median filter device");
    }
    

protected:
    FloatImage updateImpl(FloatImage img);

};

TDV_NAMESPACE_END

#endif /* TDV_MEDIANFILTERDEV_HPP */
