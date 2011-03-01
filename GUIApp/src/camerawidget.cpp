#include <QImage>
#include <QPainter>
#include <iostream>
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

CameraWidget::CameraWidget(tdv::ReadPipe<IplImage*> *framePipe)
    : m_watcher(this)
{
    m_framePipe = framePipe;
    m_lastFrame = NULL;
}

CameraWidget::~CameraWidget()
{
    m_watcher.wait();
}

void CameraWidget::init()
{
    m_watcher.start();
}

void CameraWidget::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    if ( m_lastFrame )
    {        
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

void CameraWidget::PipeWatch::run()
{
    IplImage *image;
    while ( m_self->m_framePipe->read(&image) )
    {
        m_self->m_lastFrame = image;
        m_self->update();
    }
}
