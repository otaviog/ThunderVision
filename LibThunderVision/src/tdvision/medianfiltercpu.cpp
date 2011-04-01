#include "medianfiltercpu.hpp"
#include <cv.h>

TDV_NAMESPACE_BEGIN

FloatImage MedianFilterCPU::updateImpl(FloatImage input)
{
    const Dim dim = input.dim();

    CvMat *image = input.cpuMem();
        
    FloatImage output = FloatImage::CreateCPU(dim);
    CvMat *img_output = output.cpuMem();
        
    cvSmooth(image, img_output, CV_MEDIAN);
        
    return output;
}

TDV_NAMESPACE_END
