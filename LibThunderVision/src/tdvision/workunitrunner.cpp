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

WorkUnitRunner::WorkUnitRunner(WorkUnit **wus, size_t wuCount)
{
    m_errReport = NULL;
    m_workUnits.resize(wuCount);
    std::copy(wus, wus + wuCount, m_workUnits.begin());
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
    std::cout<<ex.what()<<std::endl;
    m_errors.push_back(ex);
}

TDV_NAMESPACE_END
