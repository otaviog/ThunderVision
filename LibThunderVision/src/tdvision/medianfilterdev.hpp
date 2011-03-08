#ifndef TDV_MEDIANFILTERDEV_HPP
#define TDV_MEDIANFILTERDEV_HPP

#include <tdvbasic/common.hpp>
#include "workunit.hpp"
#include "pipe.hpp"
#include "floatimage.hpp"

TDV_NAMESPACE_BEGIN

class MedianFilterDev: public WorkUnit
{
public:        
    MedianFilterDev()       
    { 
        workName("Median filter device");
    }
    
    void input(ReadPipe<FloatImage> *rpipe)
    {
        m_rpipe = rpipe;
    }
        
    ReadPipe<FloatImage>* output()
    {
        return &m_wpipe;
    }

    bool update();

private:
    ReadPipe<FloatImage> *m_rpipe;
    ReadWritePipe<FloatImage, FloatImage> m_wpipe;
};

TDV_NAMESPACE_END

#endif /* TDV_MEDIANFILTERDEV_HPP */
