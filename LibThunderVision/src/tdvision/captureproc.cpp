#include <cv.h>
#include <highgui.h>
#include <highgui.hpp>
#include <boost/format.hpp>
#include "captureproc.hpp"

TDV_NAMESPACE_BEGIN

CaptureProc::CaptureProc(int device)
{    
    m_endCapture = false;
    m_capDevice = device;    
}

void CaptureProc::finish()
{
    m_endCapture = true;
}

void CaptureProc::process()
{
    CvCapture *capture = cvCaptureFromCAM(m_capDevice);
    
    try 
    {        
        while ( !m_endCapture )
        {
            cvGrabFrame(capture);
            IplImage *frame = cvRetrieveFrame(capture);
            
            if ( frame != NULL )
            {
                m_wpipe.write(frame);                
            }
        }
                
        cvReleaseCapture(&capture);
        capture = NULL;
        
        m_wpipe.finish();
    }
    catch (const std::exception &ex)
    {
        if ( capture != NULL )
            cvReleaseCapture(&capture);

        throw ex;
    }
}

TDV_NAMESPACE_END
