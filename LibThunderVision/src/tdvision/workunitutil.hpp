#ifndef TDV_WORKUNITUTIL_HPP
#define TDV_WORKUNITUTIL_HPP

#include "workunit.hpp"
#include "pipe.hpp"

TDV_NAMESPACE_BEGIN

template<typename InType, typename OutType = InType>
class MonoWorkUnit: public WorkUnit
{
public:
    void input(ReadPipe<InType> *rpipe)
    {
        m_rpipe = rpipe;
    }
        
    ReadPipe<OutType>* output()
    {
        return &m_wpipe;
    }
        
    bool update()
    {
        WriteGuard<ReadWritePipe<OutType> > guard(m_wpipe);
        InType data;
        if ( m_rpipe->read(&data) )
        {
            guard.write(updateImpl(data));
        }
        
        return guard.wasWrite();
    }
    
protected:
    virtual OutType updateImpl(InType data) = 0;
    
private:
    ReadPipe<InType> *m_rpipe;
    ReadWritePipe<OutType> m_wpipe;
};

template<typename InType, typename OutType = InType>
class StereoWorkUnit: public WorkUnit
{
public:
    StereoWorkUnit()
    {
        m_lrpipe = NULL;
        m_rrpipe = NULL;
    }
        
    void inputs(ReadPipe<InType> *lrpipe, ReadPipe<InType> *rrpipe)
    {
        m_rrpipe = rrpipe;
        m_lrpipe = lrpipe;
    }
        
    ReadPipe<OutType>* output()
    {
        return &m_wpipe;
    }
        
protected:       
    ReadPipe<InType> *m_lrpipe, *m_rrpipe;
    ReadWritePipe<OutType> m_wpipe;
};

TDV_NAMESPACE_END

#endif /* TDV_MISC_HPP */
