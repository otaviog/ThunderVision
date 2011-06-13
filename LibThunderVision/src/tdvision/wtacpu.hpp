#ifndef TDV_WTACPU_HPP
#define TDV_WTACPU_HPP

#include <tdvbasic/common.hpp>
#include "optimizer.hpp"

TDV_NAMESPACE_BEGIN

class WTACPU: public AbstractOptimizer
{
public:    
    WTACPU()
    {
        workName("WTACPU");
    }
    
    virtual ~WTACPU()
    {
    }

    static void wta(const Dim &dsiDim, const float *dsi, float *dispImg);
    
protected:
    void updateImpl(DSIMem mem, FloatImage img);
};

TDV_NAMESPACE_END

#endif /* TDV_WTACPU_HPP */
