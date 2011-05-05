#ifndef TDV_WORKUNITPROCESS_HPP
#define TDV_WORKUNITPROCESS_HPP

#include <tdvbasic/common.hpp>
#include <cassert>
#include <boost/thread.hpp>
#include "process.hpp"

TDV_NAMESPACE_BEGIN

class WorkUnit;

class WorkUnitProcess: public Process
{
public:
    WorkUnitProcess(WorkUnit &work)
        : m_work(work)
    {
    }
    
    void process();        
    
private:
    WorkUnit &m_work;
};

class PWorkUnitProcess: public Process
{
public:
    PWorkUnitProcess(WorkUnit *work = NULL)
    {
        m_work = work;
    }
  
    void work(WorkUnit *work)
    {
        m_work = work;
    }

    void process();
    
private:
    WorkUnit *m_work;
};

template<typename WorkUnitType>
class TWorkUnitProcess: public Process, public WorkUnitType
{
public:
    TWorkUnitProcess()
    { m_pause = false; }
    
    void pauseProc();
    
    void resumeProc();
    
    void process();
    
    void waitPauseProc();
    
private:
    boost::condition_variable m_flowCond, m_pausedCond;
    boost::mutex m_flowMutex;
    bool m_pause;            
};

template<typename WorkUnitType>
void TWorkUnitProcess<WorkUnitType>::pauseProc()
{
    boost::mutex::scoped_lock lock(m_flowMutex);
    m_pause = true;
    m_flowCond.notify_one();
}

template<typename WorkUnitType>
void TWorkUnitProcess<WorkUnitType>::resumeProc()
{
    boost::mutex::scoped_lock lock(m_flowMutex);
    m_pause = false;
    m_flowCond.notify_one();
}

template<typename WorkUnitType>
void TWorkUnitProcess<WorkUnitType>::process()
{
    bool contLoop = true;
    while ( contLoop ) 
    {        
        contLoop = WorkUnitType::update();
        
        boost::mutex::scoped_lock lock(m_flowMutex);
        if ( m_pause )
        {
            m_pausedCond.notify_one();
        }
        
        while ( m_pause )
        {
            m_flowCond.wait(lock);
        }

    }
}

template<typename WorkUnitType>
void TWorkUnitProcess<WorkUnitType>::waitPauseProc()
{
    pauseProc();
    boost::mutex::scoped_lock lock(m_flowMutex);
    while ( !m_pause )
    {
        m_pausedCond.wait(lock);
    }
}

TDV_NAMESPACE_END

#endif /* TDV_WORKUNITPROCESS_HPP */
