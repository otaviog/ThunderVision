#ifndef TDV_DYNAMICPROGCPU_HPP
#define TDV_DYNAMICPROGCPU_HPP

#include <tdvbasic/common.hpp>
#include "optimizer.hpp"

TDV_NAMESPACE_BEGIN

class SemiGlobalOptCPU: public AbstractOptimizer
{
public:
    
protected:
    void updateImpl(DSIMem mem, FloatImage img);
    
private:
};

TDV_NAMESPACE_END

#endif /* TDV_DYNAMICPROGCPU_HPP */
