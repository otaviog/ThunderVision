#ifndef TDV_BIRCHFIELDCOST_HPP
#define TDV_BIRCHFIELDCOST_HPP

#include <tdvbasic/common.hpp>
#include "matchingcost.hpp"

TDV_NAMESPACE_BEGIN

/**
 * Birchfield and Tomasi Cost Function
 */
class BirchfieldCostDev: public AbstractMatchingCost
{
public:    
    BirchfieldCostDev(int disparityMax)
        : AbstractMatchingCost(disparityMax)
    { workName("Birchfield Cost"); }
    
    Benchmark benchmark() const 
    {
        return m_benchmark;
    }
    
protected:    
    void updateImpl(FloatImage leftImg, FloatImage rightImg,
                    DSIMem dsi);
    
private:
    Benchmark m_benchmark;
};

TDV_NAMESPACE_END

#endif /* TDV_BIRCHFIELDCOST_HPP */
