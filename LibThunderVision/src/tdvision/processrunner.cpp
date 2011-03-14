#include <iostream>
#include "process.hpp"
#include "exceptionreport.hpp"
#include "processrunner.hpp"

TDV_NAMESPACE_BEGIN

struct ProcessCaller
{
public:
    ProcessCaller(ProcessRunner *runner, Process *unit)
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
    Process *m_unit;
    ProcessRunner *m_runner;
};

ProcessRunner::ProcessRunner(Process **wus, 
                             ExceptionReport *report)
{
    m_errReport = NULL;
    size_t wuCount = 0;
    
    while ( wus[wuCount] != NULL )
        wuCount++;
    
    m_workUnits.resize(wuCount);
    std::copy(wus, wus + wuCount, m_workUnits.begin());
    m_errReport = report;
    m_errorOc = false;
}

void ProcessRunner::run()
{
    for (size_t i=0; i<m_workUnits.size(); i++)
    {
        boost::thread* thread = m_threads.create_thread(ProcessCaller(this, m_workUnits[i]));
    }    
}

void ProcessRunner::reportError(const std::exception &ex)
{    
    m_errorOc = true;
    m_errReport->errorOcurred(ex);    
    for (size_t i=0; i<m_workUnits.size(); i++)
    {
        m_workUnits[i]->finish();
    }
}

TDV_NAMESPACE_END
