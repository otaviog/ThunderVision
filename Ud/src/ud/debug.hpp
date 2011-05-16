#ifndef UD_DEBUG_HPP
#define UD_DEBUG_HPP

#include <boost/format.hpp>
#include "common.hpp"

UD_NAMESPACE_BEGIN

#ifndef NDEBUG
void udDebug(const char *format, ...);
void udDebug(const boost::format &fmt);
#else
inline void udDebug(const char *format, ...) { }
inline void udDebug(const boost::format &fmt) { }
#endif

UD_NAMESPACE_END

#endif
