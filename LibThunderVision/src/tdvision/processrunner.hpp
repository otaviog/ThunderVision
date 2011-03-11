#ifndef TDV_PROCESSRUNNER_HPP
#define TDV_PROCESSRUNNER_HPP

#include <tdvbasic/common.hpp>
#include <tdvbasic/exception.hpp>
#include <boost/thread.hpp>

TDV_NAMESPACE_BEGIN

class Process;

class ProcessExceptionReport
{
public:
    virtual void errorOcurred(const std::exception &err) = 0;
    
private:
};

class ProcessRunner
{
public:
    ProcessRunner(Process **wus, ProcessExceptionReport *report);
    
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
    friend struct ProcessCaller;
    
    void reportError(const std::exception &ex);
        
    ProcessRunner(const ProcessRunner &cpy);
    
    ProcessRunner& operator=(const ProcessRunner &cpy);
    
    boost::thread_group m_threads;
    ProcessExceptionReport *m_errReport;
    std::vector<Process*> m_workUnits;
    
    bool m_errorOc;
};

TDV_NAMESPACE_END

#endif /* TDV_PROCESSRUNNER_HPP */
