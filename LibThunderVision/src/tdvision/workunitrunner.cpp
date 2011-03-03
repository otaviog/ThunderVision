#include <iostream>
#include "workunit.hpp"
#include "workunitrunner.hpp"

TDV_NAMESPACE_BEGIN

struct WorkUnitCaller
{
public:
    WorkUnitCaller(WorkUnitRunner *runner, WorkUnit *unit)
    {
        m_unit = unit;
        m_runner = runner;
    }
    
    void operator()()
    {
        try
        {
            m_unit->process();
        }
        catch (const std::exception &ex)
        {            
            m_runner->reportError(ex);
        }
    }
    
private:
    WorkUnit *m_unit;
    WorkUnitRunner *m_runner;
};

WorkUnitRunner::WorkUnitRunner(WorkUnit **wus, size_t wuCount, 
                               WorkExceptionReport *report)
{
    m_errReport = NULL;
    m_workUnits.resize(wuCount);
    std::copy(wus, wus + wuCount, m_workUnits.begin());
    m_errReport = report;
    m_errorOc = false;
}

void WorkUnitRunner::run()
{
    for (size_t i=0; i<m_workUnits.size(); i++)
    {
        boost::thread* thread = m_threads.create_thread(WorkUnitCaller(this, m_workUnits[i]));
    }    
}

void WorkUnitRunner::reportError(const std::exception &ex)
{    
    m_errorOc = true;
    m_errReport->errorOcurred(ex);    
    for (size_t i=0; i<m_workUnits.size(); i++)
    {
        m_workUnits[i]->finish();
    }
}

TDV_NAMESPACE_END
