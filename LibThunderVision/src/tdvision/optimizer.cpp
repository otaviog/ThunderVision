#include "benchmark.hpp"
#include "optimizer.hpp"

#include <iostream>

TDV_NAMESPACE_BEGIN

AbstractOptimizer::AbstractOptimizer()
{
    m_rpipe = NULL;
}

bool AbstractOptimizer::update()
{
    WriteGuard<ReadWritePipe<FloatImage> > guard(m_wpipe);
    
    DSIMem dsi;
    if ( m_rpipe->read(&dsi) )
    {
        FloatImage outimg = FloatImage::CreateDev(
            Dim(dsi.dim().width(), dsi.dim().height()));
                
        CudaBenchmarker bMarker;
        bMarker.begin();
                
        updateImpl(dsi, outimg);
        
        bMarker.end();        
        
        m_mark = bMarker.elapsedTime();
        
        std::cout << m_mark.secs() << std::endl;
        
        guard.write(outimg);
    }

    if ( !guard.wasWrite() )
    {
        finished();
    }
    
    return guard.wasWrite();
}

TDV_NAMESPACE_END
