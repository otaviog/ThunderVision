#ifndef TDV_DYNAMICPROGDEV_HPP
#define TDV_DYNAMICPROGDEV_HPP

#include <tdvbasic/common.hpp>
#include "optimizer.hpp"

TDV_NAMESPACE_BEGIN

class DynamicProgDev: public AbstractOptimizer
{
public:

protected:
    void updateImpl(DSIMem dsi, FloatImage outimg);
};
    
TDV_NAMESPACE_END

#endif /* TDV_DYNAMICPROGDEV_HPP */
