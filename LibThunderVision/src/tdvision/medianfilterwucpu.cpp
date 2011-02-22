#include "medianfilterwucpu.hpp"
#include <cv.h>

TDV_NAMESPACE_BEGIN

void MedianFilterWUCPU::process()
{
    while ( m_rpipe->waitPacket() )
    {
        FloatImageMem input = m_rpipe->read();
        
        const Dim dim = input.dim();

        IplImage *image = input.waitCPUMem();
        
        FloatImageMem output = FloatImageMem::CreateCPU(dim);
        IplImage *img_output = output.waitCPUMem();
        
        cvSmooth(image, img_output, CV_MEDIAN);
        
        m_wpipe->write(output);
        output.memRelease();
    }
}

TDV_NAMESPACE_END
