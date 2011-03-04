#include <QImage>
#include <QPainter>
#include <iostream>
#include <boost/scoped_array.hpp>
#include "camerawidget.hpp"

static QImage::Format queryQFormat(IplImage *img)
{
    QImage::Format fmt = QImage::Format_Invalid;
    
    switch (img->depth)
    {
    case IPL_DEPTH_8U:
    case IPL_DEPTH_8S:
        if ( !strcmp(img->colorModel, "RGB") )
        {
            if ( !strcmp(img->channelSeq, "BGR") )
            {
                fmt = QImage::Format_RGB888;
            }
            else
            {
                fmt = QImage::Format_RGB888;
            }
        }        
        break;
    case IPL_DEPTH_16U:
    case IPL_DEPTH_16S:
        fmt = QImage::Format_RGB16;
        break;
    case IPL_DEPTH_32S:
        fmt = QImage::Format_RGB32;
        break;
    defaut:        
        break;
    }
    
    return fmt;    
}

CameraWidget::CameraWidget(tdv::ReadPipe<IplImage*> *framePipe, bool sink)
{
    m_framePipe = framePipe;
    m_lastFrame = NULL;
    m_end = false;
    m_sink = sink;
}

CameraWidget::~CameraWidget()
{
    m_end = true;
}

void CameraWidget::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    if ( m_lastFrame != NULL )
    {            
        QMutexLocker locker(&m_imageMutex);    
        QImage::Format fmt = queryQFormat(m_lastFrame);        
        if ( fmt != QImage::Format_Invalid )
        {
            QImage img(reinterpret_cast<const uchar*>(m_lastFrame->imageData),
                       m_lastFrame->width, m_lastFrame->height,
                       m_lastFrame->widthStep, fmt);
            
            painter.drawImage(QPoint(0, 0), img.rgbSwapped());
            setFixedSize(img.width(), img.height());
        }
        else
        {
            painter.drawText(0, 0, tr("Invalid image format from camera"));
        }        
    }
}

void CameraWidget::process()
{
    if ( m_framePipe != NULL )
    {
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
}
