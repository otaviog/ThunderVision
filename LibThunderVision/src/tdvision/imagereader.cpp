#include <cv.h>
#include <highgui.h>
#include <tdvbasic/exception.hpp>
#include "imagereader.hpp"

TDV_NAMESPACE_BEGIN

bool ImageReader::update()
{
    WriteFinishGuard guard(&m_wpipe);
        
    IplImage *limg = cvLoadImage(m_filename.c_str());
    
    if ( limg != NULL )
    {
        FloatImage image(limg);
        m_wpipe.write(image);
    }
    else
    {
        throw Exception(boost::format("can't open image: %1%")
                        % m_filename.c_str());
    }

    return false;
}

TDV_NAMESPACE_END
