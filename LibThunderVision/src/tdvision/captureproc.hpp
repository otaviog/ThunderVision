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
        
    ReadPipe<IplImage*>* output()
    {
        return &m_wpipe;
    }
    
    void process();
     
    void finish();
    
private:
    ReadWritePipe<IplImage*> m_wpipe;
    bool m_endCapture;
    int m_capDevice;
};

TDV_NAMESPACE_END

#endif /* TDV_CAPTUREPROC_HPP */
