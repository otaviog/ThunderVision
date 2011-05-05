#include "matchingcost.hpp"

TDV_NAMESPACE_BEGIN

AbstractMatchingCost::AbstractMatchingCost(int disparityMax)
{
    m_lrpipe = NULL;
    m_rrpipe = NULL;
    m_maxDisparaty = disparityMax;
}

bool AbstractMatchingCost::update()
{
    WriteGuard<ReadWritePipe<DSIMem, DSIMem> > wguard(m_wpipe);
    FloatImage leftImg;
    FloatImage rightImg;

    if ( m_lrpipe->read(&leftImg) && m_rrpipe->read(&rightImg) )
    {            
        const size_t width = std::min(leftImg.dim().width(), 
                                      rightImg.dim().width());
        const size_t height = std::min(leftImg.dim().height(), 
                                       rightImg.dim().height());
        const size_t depth = m_maxDisparaty;
                            
        const Dim pktDim(width, height, depth);
        DSIMem dsi = DSIMem::Create(pktDim);            

        updateImpl(leftImg, rightImg, dsi);
        
        wguard.write(dsi);
    }
        
    return wguard.wasWrite();
}

TDV_NAMESPACE_END
