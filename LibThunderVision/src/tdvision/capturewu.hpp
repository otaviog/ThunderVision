#ifndef TDV_CAPTUREWU_HPP
#define TDV_CAPTUREWU_HPP

#include <tdvbasic/common.hpp>
#include <cv.h>
#include "floatimage.hpp"
#include "workunit.hpp"
#include "pipe.hpp"

TDV_NAMESPACE_BEGIN

class CaptureWU: public WorkUnit
{
public:
    CaptureWU(int device);
    
    ReadPipe<FloatImage>* output()
    {
        return &m_wpipe;
    }
    
    ReadPipe<IplImage*>* colorImage()
    {
        return &m_colorImagePipe;
    }
    
    void process();
     
    void finish();
    
private:
    ReadWritePipe<FloatImage, FloatImage> m_wpipe;
    ReadWritePipe<IplImage*, IplImage*> m_colorImagePipe;
    bool m_endCapture;
    int m_capDevice;
};

TDV_NAMESPACE_END

#endif /* TDV_CAPTUREWU_HPP */
