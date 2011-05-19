#ifndef TDV_MUTUALINFORMATIONDEV_HPP
#define TDV_MUTUALINFORMATIONDEV_HPP

#include <tdvbasic/common.hpp>
#include "matchingcost.hpp"

TDV_NAMESPACE_BEGIN

class MutualInformationDev: public AbstractMatchingCost
{
public:    
    MutualInformationDev(int disparityMax)
        : AbstractMatchingCost(disparityMax)
    { workName("MutualInformation"); }

protected:    
    void updateImpl(FloatImage leftImg, FloatImage rightImg,
                    DSIMem dsi);
};

TDV_NAMESPACE_END

#endif /* TDV_MUTUALINFORMATION_HPP */
