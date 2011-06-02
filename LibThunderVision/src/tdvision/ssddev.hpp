#ifndef TDV_SSDDEV_HPP
#define TDV_SSDDEV_HPP

#include <tdvbasic/common.hpp>
#include "matchingcost.hpp"

TDV_NAMESPACE_BEGIN

class SSDDev: public AbstractMatchingCost
{
public:    
    SSDDev(int disparityMax)
        : AbstractMatchingCost(disparityMax)
    { workName("SSD"); }
    
protected:    
    void updateImpl(FloatImage leftImg, FloatImage rightImg,
                    DSIMem dsi);
};

TDV_NAMESPACE_END

#endif /* TDV_SSDDEV_HPP */
