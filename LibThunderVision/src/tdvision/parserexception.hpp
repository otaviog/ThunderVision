#ifndef TDV_PARSEREXCEPTION_HPP
#define TDV_PARSEREXCEPTION_HPP

#include <tdvbasic/exception.hpp>

TDV_NAMESPACE_BEGIN

class ParserException: public Exception
{
public:
    ParserException(const std::string &msg)
        : Exception(msg)
    {
    }

    ParserException(const boost::format &format)
        : Exception(format.str())
    {
    }
};

TDV_NAMESPACE_END

#endif /* TDV_PARSEREXCEPTION_HPP */
