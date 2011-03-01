#ifndef TDV_WORKUNITRUNNER_HPP
#define TDV_WORKUNITRUNNER_HPP

#include <tdvbasic/common.hpp>
#include <tdvbasic/exception.hpp>
#include <boost/thread.hpp>

TDV_NAMESPACE_BEGIN

class WorkUnit;

class WorkUnitExceptionReport
{
public:
    virtual void errorOcurred(const std::exception &err) = 0;
    
private:
};

class WorkUnitRunner
{
public:
    WorkUnitRunner(WorkUnit **wus, size_t wuCount);
    
    void run();

    void join()
    {
        m_threads.join_all();
    }
    
    bool hasErrors() const
    {
        return !m_errors.empty();
    }
    
    size_t errorCount() const
    {
        return m_errors.size();
    } 
    
    const std::exception& error(size_t err) const
    {
        return m_errors[err];
    }
    
private:
    friend struct WorkUnitCaller;
    
    void reportError(const std::exception &ex);
        
    WorkUnitRunner(const WorkUnitRunner &cpy);
    
    WorkUnitRunner& operator=(const WorkUnitRunner &cpy);
    
    boost::thread_group m_threads;
    WorkUnitExceptionReport *m_errReport;
    std::vector<WorkUnit*> m_workUnits;
    std::vector<std::exception> m_errors;
};

TDV_NAMESPACE_END

#endif /* TDV_WORKUNITRUNNER_HPP */
