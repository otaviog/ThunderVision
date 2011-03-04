#include "iplimagesink.hpp"

TDV_NAMESPACE_BEGIN

void IplImageSink::process()
{    
    IplImage *image;
    while ( m_rpipe->read(&image) )
    {
        cvReleaseImage(&image);
    }
}

TDV_NAMESPACE_END
