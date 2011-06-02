#ifndef TDV_WTADEV_HPP
#define TDV_WTADEV_HPP

#include <tdvbasic/common.hpp>
#include "optimizer.hpp"

TDV_NAMESPACE_BEGIN

class WTADev: public AbstractOptimizer
{
public:    
    WTADev()
    {
        workName("WTA");
    }
    
    virtual ~WTADev()
    {
    }
    
    Benchmark benchmark() const
    {
        return m_mark;
    }
    
protected:
    void updateImpl(DSIMem mem, FloatImage img);
    Benchmark m_mark;
};

TDV_NAMESPACE_END

#endif /* TDV_WTADEV_HPP */
