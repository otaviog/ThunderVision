#ifndef TDV_ERRORHANDLER_HPP
#define TDV_ERRORHANDLER_HPP

#include <iostream>
#include <tdvision/processrunner.hpp>
#include <tdvision/exceptionreport.hpp>

class ErrorHandler: public tdv::ExceptionReport
{
public:
    void errorOcurred(const std::exception &err)
    {
        std::cout<<err.what()<<std::endl;
    }
};

#endif /* TDV_ERRORHANDLER_HPP */
