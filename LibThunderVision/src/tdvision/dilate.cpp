#include "sink.hpp"
#include "dilate.hpp"

TDV_NAMESPACE_BEGIN

Dilate::Dilate()
    : m_erode(CV_32F)
{
    
}

FloatImage Dilate::updateImpl(FloatImage img)
{
    FloatImage dilate = FloatImage::CreateCPU(img.dim());            
    CvMat *erode = m_erode.getImage(img.dim().width(), 
                                    img.dim().height());
        
    cvErode(img.cpuMem(), erode, NULL, 3);        
    cvDilate(erode, dilate.cpuMem(), NULL, 3);
    
    return dilate;
}

TDV_NAMESPACE_END
