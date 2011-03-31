#ifndef TDV_TEEWORKUNIT_HPP
#define TDV_TEEWORKUNIT_HPP

#include <map>
#include <tdvbasic/common.hpp>
#include "workunit.hpp"
#include "pipe.hpp"

TDV_NAMESPACE_BEGIN

template<typename TeeType>
class TeeWorkUnit: public WorkUnit
{
    typedef typename std::map<int, ReadWritePipe<TeeType>* > WPipeMap;
    
public:
    TeeWorkUnit()
    {
        m_rp = NULL;        
    }

    ~TeeWorkUnit();
    
    void input(ReadPipe<TeeType> *rpipe)
    {
        m_rp = rpipe;
    }

    void enable(int wpipeId);

    void disable(int wpipeId);

    ReadPipe<TeeType>* output(int wpipeId)
    {
        return m_wpipes[wpipeId];
    }

    bool update();

private:
    ReadPipe<TeeType> *m_rp;
    WPipeMap m_wpipes;
    std::map<int, bool> m_wpipeEnabled;
};

template<typename TeeType>
TeeWorkUnit<TeeType>::~TeeWorkUnit()
{                
    for ( typename WPipeMap::iterator mIt = m_wpipes.begin();
          mIt != m_wpipes.end(); mIt++)
    {
        delete mIt->second;
    }
}

template<typename TeeType>
bool TeeWorkUnit<TeeType>::update()
{
    TeeType data;
    const bool rd = m_rp->read(&data);

    for ( typename WPipeMap::iterator mIt = m_wpipes.begin();
          mIt != m_wpipes.end(); mIt++)
    {
        int wpId = mIt->first;
        ReadWritePipe<TeeType>* wpipe = mIt->second;

        if ( m_wpipeEnabled.count(wpId) && m_wpipeEnabled[wpId] )
        {
            if ( rd )                                 
                wpipe->write(data);
            else
                wpipe->finish();                    
        }
    }

    return rd;
}

template<typename TeeType>
void TeeWorkUnit<TeeType>::enable(int wpipeId)
{
    ReadWritePipe<TeeType> *pipe =  m_wpipes[wpipeId];
    if ( pipe == NULL )
    {
        pipe = new ReadWritePipe<TeeType>;
        m_wpipes[wpipeId] = pipe;
    }
    m_wpipeEnabled[wpipeId] = true;
}

template<typename TeeType>
void TeeWorkUnit<TeeType>::disable(int wpipeId)
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
