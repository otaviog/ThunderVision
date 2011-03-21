#ifndef TDV_WRITEEXCEPTION_HPP
#define TDV_WRITEEXCEPTION_HPP

#include <tdvbasic/common.hpp>
#include <tdvbasic/exception.hpp>

TDV_NAMESPACE_BEGIN

class WriteException: public Exception
{
public:
    WriteException(const std::string &msg)
        : Exception(msg)
    {
    }

    WriteException(const boost::format &format)
        : Exception(format.str())
    {
    }
};

TDV_NAMESPACE_END

#endif /* TDV_WRITEEXCEPTION_HPP */
