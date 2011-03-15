#ifndef TDV_EXCEPTIONREPORT_HPP
#define TDV_EXCEPTIONREPORT_HPP

#include <tdvbasic/common.hpp>
#include <exception>

TDV_NAMESPACE_BEGIN

class ExceptionReport
{
public:
    virtual void errorOcurred(const std::exception &err) = 0;
    
private:
};

TDV_NAMESPACE_END

#endif /* TDV_EXCEPTIONREPORT_HPP */
