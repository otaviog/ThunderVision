#include "imagesink.hpp"

TDV_NAMESPACE_BEGIN

void ImageSink::process()
{
    FloatImage finalImage;
    while ( m_rpipe->read(&finalImage) )
    {
        
    }
}

TDV_NAMESPACE_END
