#include <cv.h>
#include <highgui.h>
#include <tdvbasic/exception.hpp>
#include "imagereaderwu.hpp"

TDV_NAMESPACE_BEGIN

void ImageReaderWU::process()
{
    WriteFinishGuard guard(&m_wpipe);
        
    IplImage *limg = cvLoadImage(m_filename.c_str());
    
    if ( limg != NULL )
    {
        FloatImage image(limg);
        m_wpipe.write(image);
        m_wpipe.finish();
    }
    else
    {
        throw Exception(boost::format("can't open image: %1%")
                        % m_filename.c_str());
    }
}

TDV_NAMESPACE_END
