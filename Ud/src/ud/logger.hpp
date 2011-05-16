#ifndef UD_LOGGER_HPP
#define UD_LOGGER_HPP

#include <string>
#include <cstdio>
#include "common.hpp"

UD_NAMESPACE_BEGIN

class Logger
{
    static Logger m_instance;

public:
    static Logger& Get()
    {
        return m_instance;
    }

    void warnning(const char *fmt, ...);
    void debug(int line, const char *const fileName, const char *fmt, ...);
    void critical(int line, const char *const fileName, const char *fmt, ...);
    void information(const char *fmt, ...);
    void println(const char *fmt, ...);

private:
    Logger()
        : m_fileName("udlog.txt") { }
    std::string m_fileName;
};

UD_NAMESPACE_END

#endif
