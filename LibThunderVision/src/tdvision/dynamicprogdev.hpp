#ifndef TDV_DYNAMICPROGDEV_HPP
#define TDV_DYNAMICPROGDEV_HPP

#include <tdvbasic/common.hpp>
#include "optimizer.hpp"
#include "tmpcudamem.hpp"

TDV_NAMESPACE_BEGIN

class DynamicProgDev: public AbstractOptimizer
{
public:
    DynamicProgDev()
        : m_pathDSI(sizeof(int))
    {
        workName("DynamicProg");
    }
        
protected:    
    void updateImpl(DSIMem dsi, FloatImage outimg);
    
    void finished();
    
private:
    LocalDSIMem m_pathDSI;
    TmpCudaMem m_lastCostsMem;
};
    
TDV_NAMESPACE_END

#endif /* TDV_DYNAMICPROGDEV_HPP */
