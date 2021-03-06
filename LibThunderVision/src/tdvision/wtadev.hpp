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
        
protected:
    void updateImpl(DSIMem mem, FloatImage img);
};

TDV_NAMESPACE_END

#endif /* TDV_WTADEV_HPP */
