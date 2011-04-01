#include <cv.h>
#include <highgui.h>
#include <cassert>
#include <tdvbasic/util.hpp>
#include "imageresize.hpp"

TDV_NAMESPACE_BEGIN

bool ImageResize::update()
{
    FloatImage inimg;
    WriteGuard<ReadWritePipe<FloatImage> > wg(m_wpipe);
    
    if ( m_rpipe->read(&inimg) )
    {
        const Dim origDim(inimg.dim());
        
        Dim ndim(-1);
        
        switch (m_mode)
        {
        case Absolute:
            ndim = m_newDim;
            break;
        case Percent:
            ndim = Dim(float(origDim.width())*m_xpercent,
                       float(origDim.height())*m_ypercent);
            break;
        case NextPowerOf2:
            ndim = Dim(util::nextPowerOf2(origDim.width()), 
                       util::nextPowerOf2(origDim.height()));
            break;
        default:
            assert(false);
        }
        
        FloatImage outimg = FloatImage::CreateCPU(ndim);        
        cvResize(inimg.cpuMem(), outimg.cpuMem(), CV_INTER_CUBIC);
        
        wg.write(outimg);                
    }    

    return wg.wasWrite();
}

TDV_NAMESPACE_END
