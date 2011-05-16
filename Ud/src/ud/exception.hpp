#ifndef UD_EXCEPTION_HPP
#define UD_EXCEPTION_HPP

#include <exception>
#include <string>
#include <boost/format.hpp>
#include "common.hpp"

UD_NAMESPACE_BEGIN

class Exception: public std::exception
{
public:
    Exception() throw () { }

    Exception(const std::string &msg)
        : m_msg(msg) { }

    Exception(const boost::format &msg)
        : m_msg(msg.str()) { }

    virtual ~Exception() throw() { }

    const char* what() const throw()
    {
        return m_msg.c_str();
    }

private:
    std::string m_msg;
};

class LoadException: public Exception
{
public:
    LoadException(const std::string &msg)
            : Exception(msg) { }

    LoadException(const boost::format &msg)
        : Exception(msg.str()) { }

};

UD_NAMESPACE_END

#endif
