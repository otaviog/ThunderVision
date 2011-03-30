#ifndef TDV_CAPTUREPROC_HPP
#define TDV_CAPTUREPROC_HPP

#include <tdvbasic/common.hpp>
#include <cv.h>
#include <highgui.h>
#include "floatimage.hpp"
#include "process.hpp"
#include "pipe.hpp"

TDV_NAMESPACE_BEGIN

class Capture
{
public:
    Capture();
    
    void init(const std::string  &filename);
    
    void init(int device);
    
    ReadPipe<IplImage*>* output()
    {
        return &m_wpipe;
    }
    
    void update();
    
    void dispose()
    {
        if ( m_capture != NULL )
        {
            cvReleaseCapture(&m_capture);
            m_capture = NULL;
            m_wpipe.finish();
        } 
    }
    
private:
    ReadWritePipe<IplImage*> m_wpipe;
    CvCapture *m_capture;    
};


class CaptureProc: public Process
{
public:    
    CaptureProc();
    
    void init(const std::string  &filename)
    {
        m_capture.init(filename);
    }
    
    void init(int device)
    {
        m_capture.init(device);
    }

    ReadPipe<IplImage*>* output()
    {
        return m_capture.output();
    }
    
    void process();
     
    void finish();
    
private:
    Capture m_capture;
    bool m_endCapture;
};

TDV_NAMESPACE_END

#endif /* TDV_CAPTUREPROC_HPP */
