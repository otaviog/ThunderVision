#include <QImage>
#include <QPainter>
#include <iostream>
#include <boost/scoped_array.hpp>
#include <tdvision/processrunner.hpp>
#include <highgui.h>
#include "camerawidget.hpp"

CameraWidget::CameraWidget(QWidget *parent)
    : QWidget(parent)
{
    m_lastFrame = NULL;
    m_end = false;
    m_framePipe = NULL;
    m_sink = true;
    m_pixmap = NULL;
}

CameraWidget::~CameraWidget()
{
    m_end = true;

    if ( m_pixmap != NULL )
    {
        cvReleaseImage(&m_pixmap);
        m_pixmap = NULL;
    }
}

void CameraWidget::input(tdv::ReadPipe<IplImage*> *framePipe, bool sink)
{
    QMutexLocker locker(&m_imageMutex);
    m_framePipe = framePipe;
    m_sink = sink;
}

void CameraWidget::init(tdv::ExceptionReport *report)
{
    tdv::Process *procs[] = { this, NULL };

    m_procRunner = new tdv::ProcessRunner(procs, report);
    m_procRunner->run();
}

void CameraWidget::dispose()
{
    m_procRunner->join();
}

void CameraWidget::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    if ( m_lastFrame == NULL )
    {
        setMinimumSize(256, 128);
        painter.drawText(20, height()/2, tr("No image from camera"));
        return ;
    }

    {
        QMutexLocker locker(&m_imageMutex);

        if ( m_pixmap == NULL )
        {
            m_pixmap = cvCreateImage(cvGetSize(m_lastFrame), IPL_DEPTH_8U, 3);
        }
        else if ( m_pixmap->width != m_lastFrame->width
                  || m_pixmap->height != m_lastFrame->height )
        {
            cvReleaseImage(&m_pixmap);
            m_pixmap = cvCreateImage(cvGetSize(m_lastFrame), IPL_DEPTH_8U, 3);
        }
        
        cvConvertImage(m_lastFrame, m_pixmap, CV_CVTIMG_SWAP_RB);
    }

    QImage img(reinterpret_cast<const uchar*>(m_pixmap->imageData),
               m_pixmap->width, m_pixmap->height,
               m_pixmap->widthStep, QImage::Format_RGB888);

    painter.drawImage(QPoint(0, 0), img);
    setFixedSize(img.width(), img.height());
}

void CameraWidget::process()
{
    assert(m_framePipe != NULL);

    IplImage *image;
    while ( m_framePipe->read(&image) && !m_end )
    {
        QMutexLocker locker(&m_imageMutex);
        if ( m_lastFrame != NULL )
        {
            if ( m_sink )
            {
                cvReleaseImage(&m_lastFrame);
            }
        }

        m_lastFrame = image;
        update();
    }
}

IplImage* CameraWidget::lastFrame()
{
    IplImage *lfCopy = NULL;

    QMutexLocker locker(&m_imageMutex);
    if ( m_lastFrame != NULL )
    {
        lfCopy = cvCloneImage(m_lastFrame);
    }

    return lfCopy;
}
