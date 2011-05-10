#ifndef TDV_UPDATECOUNT_HPP
#define TDV_UPDATECOUNT_HPP

#include <tdvbasic/common.hpp>
#include <limits>
#include <boost/date_time/posix_time/posix_time_types.hpp>

TDV_NAMESPACE_BEGIN

class UpdateCount
{
public:
    UpdateCount()
        : m_lastTime(boost::posix_time::pos_infin)
    { 
        m_lastDuration = std::numeric_limits<float>::infinity();
        m_countPerSecs = 0.0f;
    }
    
    void count();
    
    float milliSinceCount();
    
    float lastDuration() const
    {
        return m_lastDuration;
    }
    
    float countPerSecs() const
    {
        return m_countPerSecs;
    }

    float countPerSecsNow() const;
    
private:
    boost::posix_time::ptime m_lastTime;
    float m_lastDuration, m_countPerSecs;    
};

TDV_NAMESPACE_END

#endif /* TDV_UPDATECOUNT_HPP */
