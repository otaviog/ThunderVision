#ifndef TDV_MEDIANFILTERWUCPU_HPP
#define TDV_MEDIANFILTERWUCPU_HPP

#include <tdvbasic/common.hpp>
#include "workunit.hpp"
#include "pipe.hpp"
#include "mem.hpp"

TDV_NAMESPACE_BEGIN

class MedianFilterWUCPU: public WorkUnit
{
public:
    typedef ReadPipe<FloatImageMem> ReadPipeType;
    typedef WritePipe<FloatImageMem> WritePipeType;
    
    MedianFilterWUCPU()
        : WorkUnit("Median filter CPU")
    {
        m_rpipe = NULL;
        m_wpipe = NULL;
    }
        
    void input(ReadPipeType *rpipe)
    {
        m_rpipe = rpipe;
    }

    void output(WritePipeType *wpipe)
    {
        m_wpipe = wpipe;
    }

    void process();
    
private:
    ReadPipeType *m_rpipe;
    WritePipeType *m_wpipe;
};

TDV_NAMESPACE_END

#endif /* TDV_MEDIANFILTERWUCPU_HPP */
