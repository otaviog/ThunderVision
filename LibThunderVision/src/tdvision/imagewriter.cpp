#include <cv.h>
#include <highgui.h>
#include "imagewriter.hpp"

TDV_NAMESPACE_BEGIN

bool ImageWriter::update()
{
    FloatImage fimg;
    WriteFinishGuard wg(&m_wpipe);
    
    if ( m_rpipe->read(&fimg) )
    {
        IplImage *img = fimg.cpuMem();        
        IplImage *finalImg = cvCreateImage(cvGetSize(img), 
                                           IPL_DEPTH_8U, 1);        
        cvConvertScale(img, finalImg, 255.0);
        
        cvSaveImage(m_filename.c_str(), finalImg);
        
        cvReleaseImage(&finalImg);                
        
        m_wpipe.write(fimg);
        wg.finishNotNeed();
        
        return true;
    }        
    
    return false;
}

TDV_NAMESPACE_END
