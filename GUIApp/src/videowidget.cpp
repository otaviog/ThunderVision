#include <tdvbasic/log.hpp>
#include <QImage>
#include <QMessageBox>
#include <QPainter>
#include <iostream>
#include <boost/scoped_array.hpp>
#include <tdvision/processrunner.hpp>
#include <highgui.h>
#include "videowidget.hpp"

template<typename Type, typename MatAdapter>
class VideoProcess: public tdv::Process
{
public:
    VideoProcess(VideoWidget *widget, QMutex *imageMutex, CvMat **frame)
    {        
        m_pipe = NULL;
        m_widget = widget;
        m_end = false;
        m_imgMutex = imageMutex;
        m_frame = frame;
    }
        
    void process();    

    void input(tdv::ReadPipe<Type> *pipe)
    {
        m_pipe = pipe;
    }

    void finish()
    {
        m_end = true;
    }
    
private:
    tdv::ReadPipe<Type> *m_pipe;
    Type m_lastFrame;
    VideoWidget *m_widget;
    bool m_end;
    
    QMutex *m_imgMutex;
    CvMat **m_frame;
};

template<typename Type, typename MatAdapter>
void VideoProcess<Type, MatAdapter>::process()
{
    assert(m_pipe != NULL);
    
    bool firstFrame = true;    
    Type image;
    while ( m_pipe->read(&image) && !m_end )
    {        
        QMutexLocker locker(m_imgMutex);
        if ( !firstFrame )
        {
            tdv::SinkTraits<Type>::Sinker::sink(m_lastFrame);            
        }
        
        firstFrame = false;
        m_lastFrame = image;
        
        *m_frame = MatAdapter::adapt(m_lastFrame);
        m_widget->update();
    }
}


VideoWidget::VideoWidget(QWidget *parent)
    : QWidget(parent), m_pixmap(CV_8UC3)
{
    m_matFramePipe = NULL;
    m_floatFramePipe = NULL;
    m_vidProc = NULL;
    m_lastFrame = NULL;
}

VideoWidget::~VideoWidget()
{ }

void VideoWidget::input(tdv::ReadPipe<CvMat*> *framePipe)
{
    QMutexLocker locker(&m_imageMutex);
    m_matFramePipe = framePipe;
    m_floatFramePipe = NULL;
}

void VideoWidget::input(tdv::ReadPipe<tdv::FloatImage> *framePipe)
{
    QMutexLocker locker(&m_imageMutex);
    m_floatFramePipe = framePipe;
    m_matFramePipe = NULL;
}

struct FloatAdapt
{
    static CvMat* adapt(tdv::FloatImage img)
    {
        return img.cpuMem();
    }
};

struct MatAdapt
{
    static CvMat* adapt(CvMat *img)
    {
        return img;
    }
};

void VideoWidget::init()
{    
    if ( m_matFramePipe )
    {
        VideoProcess<CvMat*, MatAdapt> *proc = 
            new VideoProcess<CvMat*, MatAdapt>(
                this, &m_imageMutex, &m_lastFrame);
        proc->input(m_matFramePipe);
        m_vidProc = proc;
    }
    else if ( m_floatFramePipe )
    {
        VideoProcess<tdv::FloatImage, FloatAdapt> *proc = 
            new VideoProcess<tdv::FloatImage, FloatAdapt>(
                this, &m_imageMutex, &m_lastFrame);
        
        proc->input(m_floatFramePipe);
        m_vidProc = proc;
    }
    
    tdv::ArrayProcessGroup grp;
    grp.addProcess(m_vidProc);
    
    m_procRunner = new tdv::ProcessRunner(grp, this);
    m_procRunner->run();
}

void VideoWidget::dispose()
{
    if ( m_vidProc != NULL )
    {
        m_vidProc->finish();
        m_procRunner->join();
    
        delete m_vidProc;
        m_vidProc = NULL;
    }
}

void VideoWidget::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    QMutexLocker locker(&m_imageMutex);
    
    if ( m_lastFrame == NULL )
    {
        setMinimumSize(256, 128);
        painter.drawText(20, height()/2, tr("No image from camera"));
        return ;
    }    

    CvMat *pixmap = m_pixmap.getImage(cvGetSize(m_lastFrame));            
    cvConvertImage(m_lastFrame, pixmap);    
    
    QImage img(reinterpret_cast<const uchar*>(pixmap->data.ptr),
               pixmap->cols, pixmap->rows,
               pixmap->step, QImage::Format_RGB888);

    painter.drawImage(QPoint(0, 0), img);
        
    setFixedSize(img.width(), img.height());
}

void VideoWidget::errorOcurred(const std::exception &err)
{
    QMetaObject::invokeMethod(this, "processError", Q_ARG(QString, tr(err.what())));
}

void VideoWidget::processError(QString msg)
{
    QMessageBox::critical(this, tr("Video input error"),
                          msg);
}

CvMat* VideoWidget::lastFrame()
{
    CvMat *lfCopy = NULL;

    QMutexLocker locker(&m_imageMutex);
    if ( m_lastFrame != NULL )
    {
        lfCopy = cvCloneMat(m_lastFrame);
    }

    return lfCopy;
}

