#include "imagesink.hpp"

TDV_NAMESPACE_BEGIN

bool ImageSink::update()
{
    FloatImage finalImage;
    if ( m_rpipe->read(&finalImage) )
    {
        finalImage.dispose();
        return true;
    }
    return false;
}

TDV_NAMESPACE_END
