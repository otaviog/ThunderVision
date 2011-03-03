#ifndef TDV_MEDIANFILTERWUCPU_HPP
#define TDV_MEDIANFILTERWUCPU_HPP

#include <tdvbasic/common.hpp>
#include "workunit.hpp"
#include "pipe.hpp"
#include "floatimage.hpp"

TDV_NAMESPACE_BEGIN

class MedianFilterWUCPU: public WorkUnit
{
public:    
    MedianFilterWUCPU()
        : WorkUnit()
    {
        workName("Median filter on CPU");
    }
        
    void input(ReadPipe<FloatImage> *rpipe)
    {
        m_rpipe = rpipe;
    }

    ReadPipe<FloatImage>* output()
    {
        return &m_wpipe;
    }

    void process();
    
private:
    ReadPipe<FloatImage> *m_rpipe;
    ReadWritePipe<FloatImage, FloatImage> m_wpipe;
};

TDV_NAMESPACE_END

#endif /* TDV_MEDIANFILTERWUCPU_HPP */
