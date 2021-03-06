#ifndef TDV_EXCEPTION_HPP
#define TDV_EXCEPTION_HPP

#include <exception>
#include <string>
#include <boost/format.hpp>
#include "common.hpp"
#include "util.hpp"

TDV_NAMESPACE_BEGIN

class Exception: public std::exception
{
public:
    Exception(const std::string &msg)
        : m_msg(msg)
    {
        util::logBacktrace();
    }

    Exception(const boost::format &format)
        : m_msg(format.str())
    {
    }

    ~Exception() throw()
    { }

    const char* what() const throw()
    {
        return m_msg.c_str();
    }

private:
    std::string m_msg;
};

TDV_NAMESPACE_END

#endif /* TDV_EXCEPTION_HPP */
