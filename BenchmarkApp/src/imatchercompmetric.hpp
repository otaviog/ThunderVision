#ifndef TDV_IMATCHERCOMPMETRIC_HPP
#define TDV_IMATCHERCOMPMETRIC_HPP

#include <tdvbasic/common.hpp>
#include <tdvision/workunit.hpp>
#include <tdvision/floatimage.hpp>

TDV_NAMESPACE_BEGIN

class IMatcherCompMetric
{
public:
    struct Report
    {
        Report()
        {
            error = 0.0;
        }
        
        double error;
    };
        
    virtual Report compare(FloatImage truthImg, FloatImage resultImg) = 0;
};

TDV_NAMESPACE_END

#endif /* TDV_IMATCHERCOMPMETRIC_HPP */
