#ifndef TDV_MEDIANFILTERCPU_HPP
#define TDV_MEDIANFILTERCPU_HPP

#include <tdvbasic/common.hpp>
#include "workunitutil.hpp"
#include "pipe.hpp"
#include "floatimage.hpp"

TDV_NAMESPACE_BEGIN

class MedianFilterCPU: public MonoWorkUnit<FloatImage, FloatImage>
{
public:    
    MedianFilterCPU()
    {
        workName("Median filter on CPU");
    }
        
protected:
    FloatImage updateImpl(FloatImage image);    
};

TDV_NAMESPACE_END

#endif /* TDV_MEDIANFILTERCPU_HPP */
