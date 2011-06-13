#include <cv.h>
#include <highgui.h>
#include <boost/format.hpp>
#include <boost/system/error_code.hpp>
#include <tdvbasic/exception.hpp>
#include <tdvbasic/util.hpp>
#include "tmpbufferimage.hpp"
#include "captureproc.hpp"

#include <iostream>

TDV_NAMESPACE_BEGIN

Capture::Capture(bool invert)
    : m_resizeTmp(CV_8UC3)
{
    m_capture = NULL;    
    m_invert = invert;
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

static void InvertImage(const CvMat *src, CvMat *dst)
{    
    for (size_t row=0; row<static_cast<size_t>(src->rows); row++)
    {
        const uchar *scolorPtr = src->data.ptr + row*src->step;
        uchar *dcolorPtr = dst->data.ptr + (src->rows - row - 1)*dst->step;
        
        for (size_t col=0; col<static_cast<size_t>(src->cols); col++)
        {            
            const uchar *scolorBase = scolorPtr + col*3;
            uchar *dcolorBase = dcolorPtr + (src->cols - col - 1)*3;
            
            for (size_t k=0; k<3; k++)
            {
                dcolorBase[k] = scolorBase[k];
            }
        }
    }
}

void Capture::update()
{
    cvGrabFrame(m_capture);
    IplImage *frame = cvRetrieveFrame(m_capture); 

    TmpBufferImage btmp(CV_8UC3);
    
    if ( frame != NULL )
    {
        CvMat *mat = cvCreateMat(frame->height, frame->width, CV_8UC3);
        if ( m_invert )
        {
            CvMat *tmp = btmp.getImage(frame->width, frame->height);        
            
            cvConvertImage(frame, tmp, CV_CVTIMG_SWAP_RB);
            InvertImage(tmp, mat);            
        }
        else
        {
            
            cvConvertImage(frame, mat, CV_CVTIMG_SWAP_RB);
        }
        
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

CaptureProc::CaptureProc(bool invert)
    : m_capture(invert)
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
