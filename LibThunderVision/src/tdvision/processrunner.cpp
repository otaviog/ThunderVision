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

ProcessRunner::ProcessRunner(ProcessGroup &procGrp, ExceptionReport *report)
    : m_procGrp(procGrp)
{
    m_errReport = report;
    m_errorOc = false;
}

void ProcessRunner::run()
{
    Process **procs = m_procGrp.processes();
    Process *proc = *procs;
    
    while ( proc != NULL )
    {
        (void) m_threads.create_thread(
            ProcessCaller(this, proc));
        proc = *++procs;
    }    
}

void ProcessRunner::reportError(const std::exception &ex)
{    
    m_errorOc = true;
    m_errReport->errorOcurred(ex);    
    Process **procs = m_procGrp.processes();
    Process *proc = *procs++;
    
    while ( proc != NULL )
    {        
        proc->finish();
        proc = *procs++;
    }
}

TDV_NAMESPACE_END
