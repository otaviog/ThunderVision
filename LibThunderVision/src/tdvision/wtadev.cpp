#include "benchmark.hpp"
#include "wtadev.hpp"

void DevWTARun(float *dsi, const tdv::Dim &dim, 
               float *outimg);

TDV_NAMESPACE_BEGIN

bool WTADev::update()
{
    WriteGuard<ReadWritePipe<FloatImage, FloatImage> > guard(m_wpipe);
    
    DSIMem dsi;
    if ( m_rpipe->read(&dsi) )
    {
        FloatImage outimg = FloatImage::CreateDev(
            Dim(dsi.dim().width(), dsi.dim().height()));
        
        float *outimg_d = outimg.devMem();
        CudaBenchmarker bm;
        bm.begin();
        
        DevWTARun(dsi.mem(), dsi.dim(), outimg_d);
        bm.end();        
        guard.write(outimg);
    }

    return guard.wasWrite();
}

TDV_NAMESPACE_END
