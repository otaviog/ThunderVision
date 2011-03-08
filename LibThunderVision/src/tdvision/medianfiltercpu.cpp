#include "medianfiltercpu.hpp"
#include <cv.h>

TDV_NAMESPACE_BEGIN

bool MedianFilterCPU::update()
{
    FloatImage input;
    WriteFinishGuard wg(&m_wpipe);
    
    if ( m_rpipe->read(&input) )
    {        
        const Dim dim = input.dim();

        IplImage *image = input.cpuMem();
        
        FloatImage output = FloatImage::CreateCPU(dim);
        IplImage *img_output = output.cpuMem();
        
        cvSmooth(image, img_output, CV_MEDIAN);
        
        m_wpipe.write(output);
        
        wg.finishNotNeed();
        return true;
    }    
    
    return false;
}

TDV_NAMESPACE_END
