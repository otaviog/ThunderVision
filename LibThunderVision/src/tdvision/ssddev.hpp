#ifndef TDV_SSDDEV_HPP
#define TDV_SSDDEV_HPP

#include <tdvbasic/common.hpp>
#include "workunit.hpp"
#include "pipe.hpp"
#include "floatimage.hpp"
#include "dsimem.hpp"

TDV_NAMESPACE_BEGIN

class SSDDev: public WorkUnit
{
public:    
    SSDDev(int disparityMax, size_t memoryByPacket);
    
    void leftImageInput(ReadPipe<FloatImage> *lpipe)
    {
        m_lrpipe = lpipe;
    }
    
    void rightImageInput(ReadPipe<FloatImage> *rpipe)
    {
        m_rrpipe = rpipe;
    }
    
    ReadPipe<DSIMem>* output()
    {
        return &m_wpipe;
    }
        
    bool update();    
    
private:
    ReadPipe<FloatImage> *m_lrpipe, *m_rrpipe;
    ReadWritePipe<DSIMem, DSIMem> m_wpipe;
    size_t m_maxDisparaty, m_memoryByPacket;
};

TDV_NAMESPACE_END

#endif /* TDV_SSDDEV_HPP */
