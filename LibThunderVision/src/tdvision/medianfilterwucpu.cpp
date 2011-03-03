#include "medianfilterwucpu.hpp"
#include <cv.h>

TDV_NAMESPACE_BEGIN

void MedianFilterWUCPU::process()
{
    FloatImage input;
    WriteFinishGuard wg(&m_wpipe);
    
    while ( m_rpipe->read(&input) )
    {        
        const Dim dim = input.dim();

        IplImage *image = input.cpuMem();
        
        FloatImage output = FloatImage::CreateCPU(dim);
        IplImage *img_output = output.cpuMem();
        
        cvSmooth(image, img_output, CV_MEDIAN);
        
        m_wpipe.write(output);
    }    
}

TDV_NAMESPACE_END
