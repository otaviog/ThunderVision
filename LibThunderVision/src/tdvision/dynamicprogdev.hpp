#ifndef TDV_DYNAMICPROGDEV_HPP
#define TDV_DYNAMICPROGDEV_HPP

#include <tdvbasic/common.hpp>
#include "optimizer.hpp"

TDV_NAMESPACE_BEGIN

class DynamicProgDev: public AbstractOptimizer
{
public:
    DynamicProgDev()
    {
        workName("DynamicProg");
    }
    
    Benchmark benchmark() const
    {
        return m_marker;
    }
    
protected:
    void updateImpl(DSIMem dsi, FloatImage outimg);

private:
    Benchmark m_marker;
};
    
TDV_NAMESPACE_END

#endif /* TDV_DYNAMICPROGDEV_HPP */
