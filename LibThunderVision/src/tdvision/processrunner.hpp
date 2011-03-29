#ifndef TDV_PROCESSRUNNER_HPP
#define TDV_PROCESSRUNNER_HPP

#include <tdvbasic/common.hpp>
#include <tdvbasic/exception.hpp>
#include <boost/thread.hpp>
#include "processgroup.hpp"

TDV_NAMESPACE_BEGIN

class Process;
class ExceptionReport;

class ProcessRunner
{
public:
    ProcessRunner(ProcessGroup &procs,
                  ExceptionReport *report);

    void run();

    void join()
    {
        m_threads.join_all();
    }
    
    void finishAll();
    
    bool errorOcurred() const
    {
        return m_errorOc;
    }
    
private:
    friend struct ProcessCaller;
    
    void reportError(const std::exception &ex);
        
    ProcessRunner(const ProcessRunner &cpy);
    
    ProcessRunner& operator=(const ProcessRunner &cpy);
    
    ArrayProcessGroup m_procGrp;

    boost::thread_group m_threads;
    ExceptionReport *m_errReport;
    bool m_errorOc;
};

TDV_NAMESPACE_END

#endif /* TDV_PROCESSRUNNER_HPP */
