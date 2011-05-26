#ifndef TDV_SEMIGLOBALDEV_HPP
#define TDV_SEMIGLOBALDEV_HPP

#include <tdvbasic/common.hpp>
#include "optimizer.hpp"

TDV_NAMESPACE_BEGIN

class SemiGlobalDev: public AbstractOptimizer
{
public:
    SemiGlobalDev()
    {
        workName("Semi-global device");
    }
    
protected:
    void updateImpl(DSIMem dsi, FloatImage outimg);
};
    
TDV_NAMESPACE_END

#endif /* TDV_DYNAMICPROGDEV_HPP */
