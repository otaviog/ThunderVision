#ifndef TDV_VIDEOWIDGET_HPP
#define TDV_VIDEOWIDGET_HPP

#include <tdvision/pipe.hpp>
#include <tdvision/process.hpp>
#include <tdvision/floatimage.hpp>
#include <tdvision/exceptionreport.hpp>
#include <tdvision/sink.hpp>
#include <tdvision/tmpbufferimage.hpp>
#include <cv.h>
#include <QWidget>
#include <QThread>
#include <QMutex>

TDV_NAMESPACE_BEGIN

class ProcessRunner;

TDV_NAMESPACE_END

class VideoWidget: public QWidget, tdv::ExceptionReport
{
    Q_OBJECT;

public:
    VideoWidget(QWidget *parent = NULL);

    ~VideoWidget();

    void input(tdv::ReadPipe<CvMat*> *framePipe);
    
    void input(tdv::ReadPipe<tdv::FloatImage> *framePipe);

    void init();

    void dispose();

    CvMat* lastFrame();
    
    void errorOcurred(const std::exception &err);
                                                
private slots:
    void processError(QString message);
    
protected:
    virtual void paintEvent(QPaintEvent *event);

private:        
    tdv::ReadPipe<CvMat*> *m_matFramePipe;
    tdv::ReadPipe<tdv::FloatImage> *m_floatFramePipe;
    
    tdv::ProcessRunner *m_procRunner;
    
    tdv::TmpBufferImage m_pixmap;
    QMutex m_imageMutex;
    CvMat *m_lastFrame;
    tdv::Process *m_vidProc;
};

#endif /* TDV_VIDEOWIDGET_HPP */
