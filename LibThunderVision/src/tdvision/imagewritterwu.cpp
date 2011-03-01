#include <cv.h>
#include <highgui.h>
#include "imagewritterwu.hpp"

TDV_NAMESPACE_BEGIN

void ImageWritterWU::process()
{
    FloatImage fimg;
    while ( m_rpipe->read(&fimg) )
    {
        IplImage *img = fimg.waitCPUMem();        
        IplImage *finalImg = cvCreateImage(cvGetSize(img), 
                                           IPL_DEPTH_8U, 1);        
        cvConvertScale(img, finalImg, 255.0);
        
        cvSaveImage(m_filename.c_str(), finalImg);
        
        cvReleaseImage(&finalImg);                
        
        m_wpipe->write(fimg);
    }
}

TDV_NAMESPACE_END
