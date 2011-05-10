#ifndef TDV_MEANQMETRIC_HPP
#define TDV_MEANQMETRIC_HPP

#include <tdvbasic/common.hpp>
#include "qualitymetric.hpp"

TDV_NAMESPACE_BEGIN

class MeanQMetric: public QualityMetric
{
public:
    Report compare(FloatImage truthImg, FloatImage resultImg);    
};

TDV_NAMESPACE_END

#endif /* TDV_MEANQMETRIC_HPP */
