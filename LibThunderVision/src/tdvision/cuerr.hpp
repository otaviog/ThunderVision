#ifndef TDV_CUERR_HPP
#define TDV_CUERR_HPP

#include <tdvbasic/common.hpp>
#include <tdvbasic/exception.hpp>
#include <cuda_runtime.h>
#include <cstdlib>

TDV_NAMESPACE_BEGIN

class CUException: public Exception
{
public:
    CUException(const std::string &msg)
        : Exception(msg)
    {
    }

    CUException(const boost::format &format)
        : Exception(format.str())
    {
    }
};

template<typename ReportPolicy>
class CUerr
{
public:
    CUerr()
    {
        m_lastErr = cudaSuccess;
    };

    CUerr(cudaError_t err)
    {
        m_lastErr = err;
    }
    
    static void checkGlobalError()
    {
        CUerr err = cudaGetLastError();
        err.checkErr();
    }

    void checkErr()
    {
        if ( m_lastErr != cudaSuccess )
        {
            ReportPolicy::report(cudaGetErrorString(m_lastErr));
        }
    }

    void checkErr(const char *line, const char *file, int linenum)
    {
        if ( m_lastErr != cudaSuccess )
        {
            ReportPolicy::report(cudaGetErrorString(m_lastErr),
                                 line, file, linenum);
        }
    }

    operator cudaError_t() const
    {
        return m_lastErr;
    }

    CUerr& operator=(cudaError_t err)
    {
        m_lastErr = err;
        return *this;
    }

    CUerr& operator<<(cudaError_t err)
    {
        m_lastErr = err;
        checkErr();
        return *this;
    }
    
    bool good() const
    {
        return m_lastErr == cudaSuccess;
    }
    
private:
    cudaError_t m_lastErr;
};

struct CUerrExpPol
{
    static void report(const char *errorStr);

    static void report(const char *errorStr, const char *line,
                       const char *file, int linenum);
};

struct CUerrLogPol
{
public:
    static void report(const char *errorStr);

    static void report(const char *errorStr, const char *line,
                       const char *file, int linenum);
};

struct CUerrLogExitPol: public CUerrLogPol
{
public:
    static void report(const char *errorStr);

    static void report(const char *errorStr, const char *line,
                       const char *file, int linenum);
};

typedef CUerr<CUerrExpPol> CUerrExp;
typedef CUerr<CUerrLogPol> CUerrLog;
typedef CUerr<CUerrLogExitPol> CUerrLogExit;

#define CUerrDB(line) { CUerrLogExit _cuerr123; _cuerr123 = line; _cuerr123.checkErr(#line, __FILE__, __LINE__); };

TDV_NAMESPACE_END

#endif /* TDV_CUERR_HPP */
