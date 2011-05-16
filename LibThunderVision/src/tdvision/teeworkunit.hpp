#ifndef TDV_TEEWORKUNIT_HPP
#define TDV_TEEWORKUNIT_HPP

#include <map>
#include <tdvbasic/common.hpp>
#include "sink.hpp"
#include "workunit.hpp"
#include "pipe.hpp"
#include "updatecount.hpp"

TDV_NAMESPACE_BEGIN

template<typename TeeType, 
         typename SinkPolicy = typename SinkTraits<TeeType>::Sinker >
class TeeWorkUnit: public WorkUnit
{
    typedef typename std::map<int, ReadWritePipe<TeeType>* > WPipeMap;
    
public:
    TeeWorkUnit()
    {
        m_rp = NULL;        
    }

    ~TeeWorkUnit();
    
    void enable(int wpipeId);

    void disable(int wpipeId);

    bool update();    
    
    float packetsBySeconds() const
    {
        return m_updateCount.countPerSecs();
    }
    
    ReadPipe<TeeType>* output(int wpipeId)
    {
        return m_wpipes[wpipeId];
    }    

    void input(ReadPipe<TeeType> *rpipe)
    {
        m_rp = rpipe;
    }
    
    ReadPipe<TeeType>* input()
    {
        return m_rp;
    }

private:
    ReadPipe<TeeType> *m_rp;
    WPipeMap m_wpipes;
    std::map<int, bool> m_wpipeEnabled;    
    UpdateCount m_updateCount;
};

template<typename TeeType, typename SinkPolicy>
TeeWorkUnit<TeeType, SinkPolicy>::~TeeWorkUnit()
{                
    for ( typename WPipeMap::iterator mIt = m_wpipes.begin();
          mIt != m_wpipes.end(); mIt++)
    {
        delete mIt->second;
    }
}

template<typename TeeType, typename SinkPolicy>
bool TeeWorkUnit<TeeType, SinkPolicy>::update()
{    
    TeeType data;
    const bool rd = m_rp->read(&data);            
    
    for ( typename WPipeMap::iterator mIt = m_wpipes.begin();
          mIt != m_wpipes.end(); mIt++)
    {
        int wpId = mIt->first;
        ReadWritePipe<TeeType> *wpipe = mIt->second;

        if ( m_wpipeEnabled.count(wpId) && m_wpipeEnabled[wpId] )
        {
            if ( rd )                                 
            {
                SinkPolicy::incrRef(data);
                wpipe->write(data);                                
            }
            else
            {
                wpipe->finish();                    
            }
        }        
    }
    
    m_updateCount.count();
    
    if ( rd )
    {
        SinkPolicy::sink(data);
    }

    return rd;
}

template<typename TeeType, typename SinkPolicy>
void TeeWorkUnit<TeeType, SinkPolicy>::enable(int wpipeId)
{
    ReadWritePipe<TeeType> *pipe =  m_wpipes[wpipeId];
    if ( pipe == NULL )
    {
        pipe = new ReadWritePipe<TeeType>;
        m_wpipes[wpipeId] = pipe;
    }
    pipe->reset();
    m_wpipeEnabled[wpipeId] = true;
}

template<typename TeeType, typename SinkPolicy>
void TeeWorkUnit<TeeType, SinkPolicy>::disable(int wpipeId)
{
    ReadWritePipe<TeeType> *pipe =  m_wpipes[wpipeId];
    if ( pipe != NULL )
    {
        pipe->finish();
        m_wpipeEnabled[wpipeId] = false;
    }
}

TDV_NAMESPACE_END

#endif /* TDV_TEEWORKUNIT_HPP */
