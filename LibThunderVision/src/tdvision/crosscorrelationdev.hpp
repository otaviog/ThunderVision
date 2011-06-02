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
    
    Benchmark benchmark() 
    {
        return m_mark;
    }
    
protected:    
    void updateImpl(FloatImage leftImg, FloatImage rightImg,
                    DSIMem dsi);
private:
    Benchmark m_mark;
};

TDV_NAMESPACE_END

#endif /* TDV_CROSSCORRELATIONDEV_HPP */
