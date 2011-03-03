#include <cv.h>
#include <highgui.h>
#include <highgui.hpp>
#include <boost/format.hpp>
#include "capturewu.hpp"

TDV_NAMESPACE_BEGIN

CaptureWU::CaptureWU(int device)
{
    workName((boost::format("Capture unit on device %1%") % device).str());
    m_endCapture = false;
    m_capDevice = device;    
}

void CaptureWU::finish()
{
    m_endCapture = true;
}

void CaptureWU::process()
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
                m_wpipe.write(FloatImage(frame));
                m_colorImagePipe.write(frame);
            }
        }
                
        cvReleaseCapture(&capture);
        capture = NULL;
        
        m_colorImagePipe.finish();
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
