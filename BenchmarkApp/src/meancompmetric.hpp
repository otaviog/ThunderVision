#ifndef TDV_MEANCOMPMETRIC_HPP
#define TDV_MEANCOMPMETRIC_HPP

#include <tdvbasic/common.hpp>
#include "imatchercompmetric.hpp"

TDV_NAMESPACE_BEGIN

class MeanCompMetric: public IMatcherCompMetric
{
public:
    Report compare(FloatImage truthImg, FloatImage resultImg);    
}

TDV_NAMESPACE_END

#endif /* TDV_MEANCOMPMETRIC_HPP */
