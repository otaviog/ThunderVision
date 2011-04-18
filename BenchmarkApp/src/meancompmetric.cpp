#include "meancompmetric.hpp"

TDV_NAMESPACE_BEGIN

static inline double dbl(double v)
{
    return v*v;
}

Report MeanCompMetric::compare(FloatImage truthImg, FloatImage resultImg)
{
    float *timg = truthImg.cpuMem();
    float *rimg = resultImg.cpuMem();
    
    size_t minSize = std::min(truthImg.dim().size(), resultImg.dim().size());
    Report report;
    
    for (size_t i=0; i<minSize; i++)
    {
        double ssd = dbl(timg[i] - rimg);        
    }
    
    return report;
}

TDV_NAMESPACE_END
