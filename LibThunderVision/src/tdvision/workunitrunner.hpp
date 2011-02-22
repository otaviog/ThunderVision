#ifndef TDV_WORKUNITRUNNER_HPP
#define TDV_WORKUNITRUNNER_HPP

#include <tdvbasic/common.hpp>
#include <boost/thread.hpp>

TDV_NAMESPACE_BEGIN

class WorkUnit;

class WorkUnitRunner
{
public:
    WorkUnitRunner()
    {
    }
    
    void run(WorkUnit **wus, size_t wuCount);

    void join()
    {
        m_threads.join_all();
    }
private:
    boost::thread_group m_threads;
};

TDV_NAMESPACE_END

#endif /* TDV_WORKUNITRUNNER_HPP */
