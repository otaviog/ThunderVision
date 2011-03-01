#ifndef TDV_SSDWUDEV_HPP
#define TDV_SSDWUDEV_HPP

#include <tdvbasic/common.hpp>
#include "workunit.hpp"
#include "pipe.hpp"
#include "floatimage.hpp"
#include "dsimem.hpp"

TDV_NAMESPACE_BEGIN

class SSDWUDev: public WorkUnit
{
public:    
    SSDWUDev(int disparityMax, size_t memoryByPacket);
    
    void leftImageInput(ReadPipe<FloatImage> *lpipe)
    {
        m_lrpipe = lpipe;
    }
    
    void rightImageInput(ReadPipe<FloatImage> *rpipe)
    {
        m_rrpipe = rpipe;
    }
    
    void output(WritePipe<DSIMem> *wpipe)
    {
        m_wpipe = wpipe;
    }
        
    void process();    
    
private:
    ReadPipe<FloatImage> *m_lrpipe, *m_rrpipe;
    WritePipe<DSIMem> *m_wpipe;
    size_t m_maxDisparaty, m_memoryByPacket;
};

TDV_NAMESPACE_END

#endif /* TDV_SSDWUDEV_HPP */
