#include <cstdarg>
#include <cstdio>
#include <iostream>
#include "debug.hpp"

using std::FILE;
using std::va_list;
using std::vfprintf;

UD_NAMESPACE_BEGIN

#ifndef NDEBUG

static FILE *out_debug = stdout;

void udDebug(const char *format, ...)
{
    va_list lst;
    va_start(lst, format);
    vfprintf(out_debug, format, lst);
}

void udDebug(const boost::format &fmt)
{
    std::cout<<fmt.str()<<std::endl;
}
#endif

UD_NAMESPACE_END
