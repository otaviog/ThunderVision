#ifndef TDV_SEMIGLOBALDEV_HPP
#define TDV_SEMIGLOBALDEV_HPP

#include <tdvbasic/common.hpp>
#include "optimizer.hpp"
#include "semiglobal.h"

TDV_NAMESPACE_BEGIN

class SemiGlobalDev: public AbstractOptimizer
{
public:
    SemiGlobalDev()
    {
        workName("Semi-global device");
        m_zeroAggregDSI = true;
    }
    
protected:
    void updateImpl(DSIMem dsi, FloatImage outimg);
    
    void finished();
    
private:
    LocalDSIMem m_aggregDSI;
    bool m_zeroAggregDSI;    
    SGPaths m_sgPaths;
};
    
TDV_NAMESPACE_END

#endif /* TDV_DYNAMICPROGDEV_HPP */
