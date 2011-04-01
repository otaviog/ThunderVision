#include "rgbconv.hpp"

TDV_NAMESPACE_BEGIN

RGBConv::RGBConv()
{
    workName("RGB converter");
}

CvMat* RGBConv::updateImpl(FloatImage image)
{
    CvMat *img = image.cpuMem();
    CvMat *scaleImg = cvCreateMat(img->height, img->width, CV_8U);
    CvMat *imgRGB = cvCreateMat(img->height, img->width, CV_8UC3);
        
    cvConvertScale(img, scaleImg, 255.0);

    cvCvtColor(scaleImg, imgRGB, CV_GRAY2RGB);
    cvReleaseMat(&scaleImg);
        
    image.dispose();
    
    return imgRGB;        
}

TDV_NAMESPACE_END
