#ifndef TDV_CAMERAWIDGET_HPP
#define TDV_CAMERAWIDGET_HPP

#include <tdvision/pipe.hpp>
#include <tdvision/process.hpp>
#include <tdvision/floatimage.hpp>
#include <tdvision/exceptionreport.hpp>
#include <cv.h>
#include <QWidget>
#include <QThread>
#include <QMutex>

class CameraWidget: public QWidget, public tdv::Process
{
    Q_OBJECT;

public:
    CameraWidget();

    ~CameraWidget();

    void input(tdv::ReadPipe<IplImage*> *framePipe, bool sink);

    void init(tdv::ExceptionReport *report);

    void shutdown();

    IplImage* lastFrame();

    void process();

protected:
    virtual void paintEvent(QPaintEvent *event);

private:
    tdv::ReadPipe<IplImage*> *m_framePipe;
    tdv::ProcessRunner *m_procRunner;
    
    IplImage *m_lastFrame;
    bool m_end, m_sink;

    QMutex m_imageMutex;
};

#endif /* TDV_CAMERAWIDGET_HPP */
