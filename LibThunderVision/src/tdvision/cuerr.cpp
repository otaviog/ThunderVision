#include <tdvbasic/log.hpp>
#include "cuerr.hpp"

TDV_NAMESPACE_BEGIN

void CUerrExpPol::report(const char *errorStr)
{    
    throw CUException(errorStr);
}

void CUerrExpPol::report(const char *errorStr, const char *line,
                         const char *file, int linenum)
{
    throw CUException(boost::format("%1%:%2%:%3% - %4%")
                      % file % linenum % line % errorStr);
}

void CUerrLogPol::report(const char *errorStr)
{
    TDV_LOG(warn).printf("%s", errorStr);
}

void CUerrLogPol::report(const char *errorStr, const char *line,
                         const char *file, int linenum)
{
    TDV_LOG(warn).printf("%s:%d:%s - %s",
                         file, linenum, line, errorStr);
}

void CUerrLogExitPol::report(const char *errorStr)
{
    TDV_LOG(fatal).printf("%s", errorStr);
    exit(1);
}

void CUerrLogExitPol::report(const char *errorStr, const char *line,
                             const char *file, int linenum)
{
    TDV_LOG(fatal).printf("%s:%d:%s - %s",
                          file, linenum, line, errorStr);
    exit(1);
}


TDV_NAMESPACE_END
