#ifndef TDV_WORKUNITPROCESS_HPP
#define TDV_WORKUNITPROCESS_HPP

#include <tdvbasic/common.hpp>
#include <cassert>
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
    { }
    
    void process()
    {
        while ( WorkUnitType::update() );
    }    
    
private:
};

TDV_NAMESPACE_END

#endif /* TDV_WORKUNITPROCESS_HPP */
