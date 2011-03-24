#ifndef TDV_TEEWORKUNIT_HPP
#define TDV_TEEWORKUNIT_HPP

#include <tdvbasic/common.hpp>
#include "workunit.hpp"

TDV_NAMESPACE_BEGIN

template<typename TeeType>
class TeeWorkUnit: public WorkUnit
{
public:
    TeeWorkUnit()
    {
        m_rp = NULL;
        m_enableOut2 = false;
    }
    
    void input(ReadPipe<TeeType> *rpipe)
    {
        m_rp = rpipe;
    }
    
    void enableOutput2()
    {
        m_enableOut2 = true;
    }
    
    void disableOutput2()
    {
        m_enableOut2 = false;
    }

    ReadPipe<TeeType> output1()
    {
        return &m_wp1;
    }
    
    ReadPipe<TeeType> output2()
    {
        return &m_wp2;
    }
    
    bool update()
    {        
        TeeType data;
        if ( m_rp->read(&data) )
        {
            m_wp1.write(data);
            if ( m_enableOut2 )
                m_wp2.write(data);
            return true;
        }
        
        return false;
    }
    
private:
    tdv::ReadPipe<TeeType> *m_rp;
    tdv::ReadWritePipe<TeeType> m_wp1;
    tdv::ReadWritePipe<TeeType> m_wp2;
    bool m_enableOut2;
}

TDV_NAMESPACE_END

#endif /* TDV_TEEWORKUNIT_HPP */
