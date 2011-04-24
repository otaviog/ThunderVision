#include "meancompmetric.hpp"

TDV_NAMESPACE_BEGIN

static inline double dbl(double v)
{
    return v*v;
}

IMatcherCompMetric::Report MeanCompMetric::compare(
    FloatImage truthImg, FloatImage resultImg)
{
    float * const timg = truthImg.cpuMem()->data.fl;
    float * const rimg = resultImg.cpuMem()->data.fl;
    

    size_t minSize = std::min(truthImg.dim().size(), resultImg.dim().size());
    Report report;
    
    long double errorMean = 0.0;
    
    for (size_t i=0; i<minSize; i++)
    {
        double sad = std::abs(timg[i] - rimg[i]);
        errorMean += sad;
    }
    
    errorMean /= double(truthImg.dim().size());
    report.error = errorMean;
    
    return report;
}

TDV_NAMESPACE_END
