#ifndef TDV_SSDDEV_HPP
#define TDV_SSDDEV_HPP

#include <tdvbasic/common.hpp>
#include "matchingcost.hpp"

TDV_NAMESPACE_BEGIN

class SSDDev: public MatchingCost
{
public:    
    SSDDev(int disparityMax);
    
    void inputs(ReadPipe<FloatImage> *lpipe, ReadPipe<FloatImage> *rpipe)
    {
        m_lrpipe = lpipe;
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
