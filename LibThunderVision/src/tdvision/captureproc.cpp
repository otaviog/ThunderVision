#include <cv.h>
#include <highgui.h>
#include <boost/format.hpp>
#include <boost/system/error_code.hpp>
#include <tdvbasic/exception.hpp>
#include "captureproc.hpp"

TDV_NAMESPACE_BEGIN

Capture::Capture()
{
    m_capture = NULL;
}

void Capture::init(const std::string &filename)
{
    m_capture = cvCaptureFromFile(filename.c_str());
    if ( m_capture == NULL )
    {        
        boost::system::error_code errcode;
        throw Exception(boost::format("Can't open file %1%: %2%") 
                        % filename % errcode.message());
    }
}

void Capture::init(int capDevice)
{
    m_capture = cvCaptureFromCAM(capDevice);
    if ( m_capture == NULL )
    {
        throw Exception(boost::format("Can't open capture device %1%") 
                        % capDevice);
    }
}

void Capture::update()
{
    int frameCount = static_cast<int>(
        cvGetCaptureProperty(m_capture, CV_CAP_PROP_FRAME_COUNT));
    
    int nxtFrame = static_cast<int>(
        cvGetCaptureProperty(m_capture, CV_CAP_PROP_POS_FRAMES));
    
    if ( nxtFrame >= frameCount )
    {
        m_wpipe.finish();
        return ;
    }
    
    cvGrabFrame(m_capture);        
    IplImage *frame = cvRetrieveFrame(m_capture);
        
    if ( frame != NULL )
    {
        CvMat *mat = cvCreateMat(frame->height, frame->width, CV_8UC3);
        cvConvertImage(frame, mat, CV_CVTIMG_SWAP_RB);
        m_wpipe.write(mat);
    }    
}

void Capture::dispose()
{
    if ( m_capture != NULL )
    {
        cvReleaseCapture(&m_capture);
        m_capture = NULL;
        m_wpipe.finish();
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
