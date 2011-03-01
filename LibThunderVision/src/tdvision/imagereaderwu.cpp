#include <cv.h>
#include <highgui.h>
#include "imagereaderwu.hpp"

TDV_NAMESPACE_BEGIN

void ImageReaderWU::process()
{
    FloatImage image(cvLoadImage(m_filename.c_str()));
    m_wpipe->write(image);
}

TDV_NAMESPACE_END
