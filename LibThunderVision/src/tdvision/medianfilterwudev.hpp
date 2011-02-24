#ifndef TDV_MEDIANFILTERWUDEV_HPP
#define TDV_MEDIANFILTERWUDEV_HPP

#include <tdvbasic/common.hpp>
#include "workunit.hpp"
#include "pipe.hpp"
#include "floatimage.hpp"

TDV_NAMESPACE_BEGIN

class MedianFilterWUDev: public WorkUnit
{
public:    
    typedef ReadPipe<FloatImage> ReadPipeType;
    typedef WritePipe<FloatImage> WritePipeType;
    
    MedianFilterWUDev()
        : WorkUnit("Median filter device")
    { }
    
    virtual void input(ReadPipeType *rpipe)
    {
        m_rpipe = rpipe;
    }
        
    virtual void output(WritePipeType *wpipe)
    {
        m_wpipe = wpipe;
    }

    void process();

private:
    ReadPipeType *m_rpipe;
    WritePipeType *m_wpipe;
};

TDV_NAMESPACE_END

#endif /* TDV_MEDIANFILTERWUDEV_HPP */
