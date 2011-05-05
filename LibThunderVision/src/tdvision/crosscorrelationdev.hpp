#ifndef TDV_CROSSCORRELATIONDEV_HPP
#define TDV_CROSSCORRELATIONDEV_HPP

#include <tdvbasic/common.hpp>
#include "matchingcost.hpp"

TDV_NAMESPACE_BEGIN

class CrossCorrelationDev: public AbstractMatchingCost
{
public:
    CrossCorrelationDev(int disparityMax)
        : AbstractMatchingCost(disparityMax)
    { workName("CrossCorrelation"); }

protected:    
    void updateImpl(FloatImage leftImg, FloatImage rightImg,
                    DSIMem dsi);

};

TDV_NAMESPACE_END

#endif /* TDV_CROSSCORRELATIONDEV_HPP */
