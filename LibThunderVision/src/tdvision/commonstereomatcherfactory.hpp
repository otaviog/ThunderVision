#ifndef TDV_COMMONSTEREOMATCHERFACTORY_HPP
#define TDV_COMMONSTEREOMATCHERFACTORY_HPP

#include <tdvbasic/common.hpp>
#include "stereomatcherfactory.hpp"

TDV_NAMESPACE_BEGIN

class CommonStereoMatcherFactory: public StereoMatcherFactory
{
public:        
    enum ComputeDevMode
    {
        Device, CPU
    };

    enum MatchMode
    {
        SSD, CrossCorrelationNorm
    };

    enum OptMode
    {
        WTA, DynamicProg, DynamicProgOnCPU, Global
    };
        
    CommonStereoMatcherFactory();
    
    StereoMatcher* createStereoMatcher();
    
    void computeDev(ComputeDevMode compMode)
    {
        m_compMode = compMode;
    }
    
    void matchingCost(MatchMode mtmd)
    {
        m_matchMode = mtmd;
    }
    
    void optimization(OptMode optmd)
    {
        m_optMode = optmd;
    }
    
    void maxDisparity(int value)
    {
        m_maxDisparity = value;
    }
    
private:
    ComputeDevMode m_compMode;
    MatchMode m_matchMode;
    OptMode m_optMode;
    int m_maxDisparity;
};

TDV_NAMESPACE_END

#endif /* TDV_COMMONSTEREOMATCHERFACTORY_HPP */
