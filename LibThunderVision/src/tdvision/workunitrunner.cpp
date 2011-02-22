#include "workunit.hpp"
#include "workunitrunner.hpp"

TDV_NAMESPACE_BEGIN

struct WorkUnitCaller
{
public:
    WorkUnitCaller(WorkUnit *unit)
    {
        m_unit = unit;
    }
    
    void operator()()
    {
        m_unit->process();
    }
    
private:
    WorkUnit *m_unit;
};

void WorkUnitRunner::run(WorkUnit **wus, size_t wuCount)
{
    for (size_t i=0; i<wuCount; i++)
    {
        boost::thread* thread = m_threads.create_thread(WorkUnitCaller(wus[i]));
    }    
}

TDV_NAMESPACE_END
