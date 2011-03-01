#ifndef TDV_CAPTUREWU_HPP
#define TDV_CAPTUREWU_HPP

#include <tdvbasic/common.hpp>
#include <cv.h>
#include "floatimage.hpp"
#include "typedworkunit.hpp"
#include "pipe.hpp"

TDV_NAMESPACE_BEGIN

class CaptureWU: public TypedWorkUnit<FloatImage, FloatImage>
{
public:
    CaptureWU(int device);
    
    void input(ReadPipeType *rpipe)
    { }

    void output(WritePipeType *wpipe)
    {
        m_wpipe = wpipe;
    }
    
    void colorImage(WritePipe<IplImage*> *pipe)
    {
        m_colorImagePipe = pipe;
    }
    
    void process();
     
    void endCapture();
    
private:
    WritePipe<FloatImage> *m_wpipe;
    WritePipe<IplImage*> *m_colorImagePipe;
    bool m_endCapture;
    int m_capDevice;
};

TDV_NAMESPACE_END

#endif /* TDV_CAPTUREWU_HPP */
