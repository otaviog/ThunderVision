#include <cstdarg>
#include <boost/thread/mutex.hpp>
#include <cstdio>
#include "logger.hpp"

UD_NAMESPACE_BEGIN

using namespace std;

class ScopedFile
{
public:
    ScopedFile()
    {
        m_file = NULL;
    }

    ~ScopedFile()
    {
        close();
    }

    FILE* open(const std::string &filename, const char *mode)
    {
        m_file = fopen(filename.c_str(), mode);
		
        if ( !m_file )
            perror("Can't create the log file.");

        return m_file;
    }

    void close()
    {
        if ( NULL != m_file )
            fclose(m_file);
    }

private:
    FILE* m_file;
};

Logger Logger::m_instance;
static boost::mutex fileMutex;

void Logger::warnning(const char *fmt, ...)
{
    boost::mutex::scoped_lock slk(fileMutex);
    ScopedFile file;

    FILE *out = file.open(m_fileName, "a");

    if ( NULL == out )
        return ;

    va_list ap;
    va_start(ap, fmt);
    fprintf(out, "warnning: ");
    vfprintf(out, fmt, ap);
    fprintf(out, "\n");
    va_end(ap);
}

void Logger::debug(int line, const char *const fileName, const char *fmt, ...)
{
    boost::mutex::scoped_lock slk(fileMutex);
    ScopedFile file;
    FILE *out = file.open(m_fileName, "a");

    if ( NULL == out )
        return ;

    va_list ap;
    va_start(ap, fmt);
    fprintf(out, "debug: ");
    vfprintf(out, fmt, ap);
    fprintf(out, " at %d in %s\n", line, fileName);
    va_end(ap);
}

void Logger::critical(int line, const char *const fileName, const char *fmt, ...)
{
    boost::mutex::scoped_lock slk(fileMutex);
    ScopedFile file;
    FILE *out = file.open(m_fileName, "a");

    if ( NULL == out )
        return ;

    va_list ap;
    va_start(ap, fmt);
    fprintf(out, "debug: ");
    vfprintf(out, fmt, ap);
    fprintf(out, " at %d in %s\n", line, fileName);
    va_end(ap);
}

void Logger::information(const char *fmt, ...)
{
    boost::mutex::scoped_lock slk(fileMutex);
    ScopedFile file;
    FILE *out = file.open(m_fileName, "a");

    if ( NULL == out )
        return ;

    va_list ap;
    va_start(ap, fmt);
    fprintf(out, "information: ");
    vfprintf(out, fmt, ap);
    fprintf(out, "\n");
    va_end(ap);
}

void Logger::println(const char *fmt, ...)
{
    boost::mutex::scoped_lock slk(fileMutex);
    ScopedFile file;
    FILE *out = file.open(m_fileName, "a");

    if ( NULL == out )
        return ;

    va_list ap;
    va_start(ap, fmt);
    vfprintf(out, fmt, ap);
    fprintf(out, "\n");
    va_end(ap);
}

UD_NAMESPACE_END
