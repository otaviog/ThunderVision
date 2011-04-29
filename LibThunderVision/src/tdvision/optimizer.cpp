#include "benchmark.hpp"
#include "optimizer.hpp"

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
                
        CudaBenchmarker bm;
        bm.begin();
        
        updateImpl(dsi, outimg);
        
        bm.end();        
        guard.write(outimg);
    }

    return guard.wasWrite();

}

TDV_NAMESPACE_END
