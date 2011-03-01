#include <cstdarg>
#include <cstdio>
#include "log.hpp"

TDV_NAMESPACE_BEGIN

Log g_tdvLog;

const size_t LogCategory::MAX_MESSAGE_SIZE = 500;

char message[500];
void LogCategory::printf(const char *format, ...)
{
    va_list ap;
    va_start(ap, format);

    char message[MAX_MESSAGE_SIZE];
    size_t size = vsnprintf(message, MAX_MESSAGE_SIZE, format, ap);
    va_end(ap);

    assert(size >= 0);

    emitMessage(message, size);
}

void TdvGlobalLogDefaultOutputs()
{
    boost::shared_ptr<StdErrLogOutput> errOutput(new StdErrLogOutput);
    g_tdvLog.registerOutput("deb", errOutput);
    g_tdvLog.registerOutput("warn", errOutput);
    g_tdvLog.registerOutput("fatal", errOutput);
}

TDV_NAMESPACE_END
