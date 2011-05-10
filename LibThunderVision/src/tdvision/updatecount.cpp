#include "updatecount.hpp"

TDV_NAMESPACE_BEGIN

namespace pxt = boost::posix_time;

float UpdateCount::milliSinceCount()
{
    const pxt::ptime time = 
        pxt::microsec_clock::local_time();
    
    if ( !m_lastTime.is_infinity() )
    {
        const pxt::time_duration td = time - m_lastTime;        
        return td.total_milliseconds();
    }
    else
    {
        return 0.0f;    
    }
}

void UpdateCount::count()
{        
    const pxt::ptime time = 
        pxt::microsec_clock::local_time();
    
    if ( !m_lastTime.is_infinity() )
    {
        const pxt::time_duration td = time - m_lastTime;        
        m_lastDuration = td.total_milliseconds();
        m_countPerSecs = 1000.0/m_lastDuration;
    }
    
    m_lastTime = time;

}

float UpdateCount::countPerSecsNow() const
{
    const pxt::ptime time = 
        pxt::microsec_clock::local_time();
    
    if ( !m_lastTime.is_infinity() )
    {
        const pxt::time_duration td = time - m_lastTime;        
        float duration = td.total_milliseconds();
        float countPerSecs = 1000.0/duration;
        return countPerSecs;
    }
    else
    {
        return 0.0f;
    }
}

TDV_NAMESPACE_END
