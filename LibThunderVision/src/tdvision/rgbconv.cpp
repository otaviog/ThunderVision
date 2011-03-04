#include "rgbconv.hpp"

TDV_NAMESPACE_BEGIN

RGBConv::RGBConv()
{
    workName("RGB converter");
}

void RGBConv::process()
{
    WriteFinishGuard wguard(&m_wpipe);
    
    FloatImage image;
    while ( m_rpipe->read(&image) )
    {
        IplImage *img = image.cpuMem();
        IplImage *scaleImg = cvCreateImage(cvGetSize(img), 
                                           IPL_DEPTH_8U, 1);
        IplImage *imgRGB = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3);
        
        cvConvertScale(img, scaleImg, 255.0);

        cvCvtColor(scaleImg, imgRGB, CV_GRAY2RGB);
        cvReleaseImage(&scaleImg);
        
        image.dispose();
        
        m_wpipe.write(imgRGB);
    }    
}

TDV_NAMESPACE_END
