#ifndef TDV_CAPTUREPROC_HPP
#define TDV_CAPTUREPROC_HPP

#include <tdvbasic/common.hpp>
#include <cv.h>
#include "floatimage.hpp"
#include "process.hpp"
#include "pipe.hpp"

TDV_NAMESPACE_BEGIN

class CaptureProc: public Process
{
public:
    CaptureProc(int device);
    
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

#endif /* TDV_CAPTUREPROC_HPP */
