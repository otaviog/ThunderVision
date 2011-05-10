#ifndef TDV_QUALITYMETRIC_HPP
#define TDV_QUALITYMETRIC_HPP

#include <tdvbasic/common.hpp>
#include <tdvision/workunit.hpp>
#include <tdvision/floatimage.hpp>

TDV_NAMESPACE_BEGIN

class QualityMetric
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

#endif /* TDV_QUALITYMETRIC_HPP */
