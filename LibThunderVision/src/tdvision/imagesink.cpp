#include "imagesink.hpp"

TDV_NAMESPACE_BEGIN

bool ImageSink::update()
{
    FloatImage finalImage;
    return m_rpipe->read(&finalImage);
}

TDV_NAMESPACE_END
