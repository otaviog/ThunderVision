#include <cv.h>
#include <highgui.h>
#include <highgui.hpp>
#include <boost/format.hpp>
#include "capturewu.hpp"

TDV_NAMESPACE_BEGIN

CaptureWU::CaptureWU(int device)
    : WorkUnit((boost::format("Capture unit on device %1%") % device).str())
{
    m_endCapture = false;
    m_capDevice = device;
}

void CaptureWU::process()
{
    cv::VideoCapture capture(m_capDevice);    
    cv::Mat frame;
    
    while ( !m_endCapture )
    {
        capture >> frame;               
        m_wpipe->write(FloatImage(&frame));
    }
}

TDV_NAMESPACE_END
