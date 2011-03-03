#ifndef TDV_WORKUNITRUNNER_HPP
#define TDV_WORKUNITRUNNER_HPP

#include <tdvbasic/common.hpp>
#include <tdvbasic/exception.hpp>
#include <boost/thread.hpp>

TDV_NAMESPACE_BEGIN

class WorkUnit;

class WorkExceptionReport
{
public:
    virtual void errorOcurred(const std::exception &err) = 0;
    
private:
};

class WorkUnitRunner
{
public:
    WorkUnitRunner(WorkUnit **wus, size_t wuCount, 
                   WorkExceptionReport *report);
    
    void run();

    void join()
    {
        m_threads.join_all();
    }
    
    bool errorOcurred() const
    {
        m_errorOc;
    }
private:
    friend struct WorkUnitCaller;
    
    void reportError(const std::exception &ex);
        
    WorkUnitRunner(const WorkUnitRunner &cpy);
    
    WorkUnitRunner& operator=(const WorkUnitRunner &cpy);
    
    boost::thread_group m_threads;
    WorkExceptionReport *m_errReport;
    std::vector<WorkUnit*> m_workUnits;
    
    bool m_errorOc;
};

TDV_NAMESPACE_END

#endif /* TDV_WORKUNITRUNNER_HPP */
