#ifndef TDV_MEDIANFILTERCPU_HPP
#define TDV_MEDIANFILTERCPU_HPP

#include <tdvbasic/common.hpp>
#include "workunit.hpp"
#include "pipe.hpp"
#include "floatimage.hpp"

TDV_NAMESPACE_BEGIN

class MedianFilterCPU: public WorkUnit
{
public:    
    MedianFilterCPU()
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

    bool update();
    
private:
    ReadPipe<FloatImage> *m_rpipe;
    ReadWritePipe<FloatImage, FloatImage> m_wpipe;
};

TDV_NAMESPACE_END

#endif /* TDV_MEDIANFILTERCPU_HPP */
