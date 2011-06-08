#include "matchingcost.hpp"

#include <iostream>

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
        
        if ( m_dsi.dim().size() != pktDim.size() )
        {
            m_dsi = DSIMem();
            m_dsi = DSIMem::Create(pktDim, leftImg);
        }
        
        m_dsi.leftOrigin(leftImg);
        
        CudaBenchmarker bMarker;       
        bMarker.begin();
        
        updateImpl(leftImg, rightImg, m_dsi);
        
        bMarker.end();
        
        //m_mark.addProbe(bMarker.elapsedTime());
        m_mark = bMarker.elapsedTime();
        std::cout << m_mark.secs() << std::endl;
        
        wguard.write(m_dsi);
    }
        
    return wguard.wasWrite();
}

TDV_NAMESPACE_END
