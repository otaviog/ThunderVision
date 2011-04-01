#ifndef TDV_VIDEOWIDGET_HPP
#define TDV_VIDEOWIDGET_HPP

#include <tdvision/pipe.hpp>
#include <tdvision/process.hpp>
#include <tdvision/floatimage.hpp>
#include <tdvision/exceptionreport.hpp>
#include <tdvision/sink.hpp>
#include <cv.h>
#include <QWidget>
#include <QThread>
#include <QMutex>

TDV_NAMESPACE_BEGIN

class ProcessRunner;

TDV_NAMESPACE_END

class VideoWidget: public QWidget, public tdv::Process, tdv::ExceptionReport
{
    Q_OBJECT;

public:
    VideoWidget(QWidget *parent = NULL);

    ~VideoWidget();

    void input(tdv::ReadPipe<CvMat*> *framePipe);

    void init();

    void dispose();

    CvMat* lastFrame();

    void process();
    
    void errorOcurred(const std::exception &err);

private slots:
    void processError(QString message);
    
protected:
    virtual void paintEvent(QPaintEvent *event);

private:
    tdv::ReadPipe<CvMat*> *m_framePipe;
    tdv::ProcessRunner *m_procRunner;
    
    IplImage *m_pixmap;
    CvMat *m_lastFrame;
    bool m_end;

    QMutex m_imageMutex;
};

#endif /* TDV_VIDEOWIDGET_HPP */
