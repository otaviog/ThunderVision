#include <cv.h>
#include <highgui.h>
#include <boost/format.hpp>
#include "captureproc.hpp"

TDV_NAMESPACE_BEGIN

Capture::Capture()
{
    m_capture = NULL;
}

void Capture::init(const std::string &filename)
{
    m_capture = cvCaptureFromFile(filename.c_str());
}

void Capture::init(int capDevice)
{
    m_capture = cvCaptureFromCAM(capDevice);
}

void Capture::update()
{
    cvGrabFrame(m_capture);
    IplImage *frame = cvRetrieveFrame(m_capture);
            
    if ( frame != NULL )
    {
        m_wpipe.write(frame);                
    }    
}

CaptureProc::CaptureProc()
{    
    m_endCapture = false;    
}

void CaptureProc::finish()
{
    m_endCapture = true;
}

void CaptureProc::process()
{        
    try 
    {        
        while ( !m_endCapture )
        {
            m_capture.update();
        }
                
        m_capture.dispose();
    }
    catch (const std::exception &ex)
    {
        m_capture.dispose();
        throw ex;
    }

}

TDV_NAMESPACE_END
