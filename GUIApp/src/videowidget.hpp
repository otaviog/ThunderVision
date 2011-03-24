#ifndef TDV_VIDEOWIDGET_HPP
#define TDV_VIDEOWIDGET_HPP

#include <tdvision/pipe.hpp>
#include <tdvision/process.hpp>
#include <tdvision/floatimage.hpp>
#include <tdvision/exceptionreport.hpp>
#include <cv.h>
#include <QWidget>
#include <QThread>
#include <QMutex>

TDV_NAMESPACE_BEGIN

class ProcessRunner;

TDV_NAMESPACE_END

class VideoWidget: public QWidget, public tdv::Process
{
    Q_OBJECT;

public:
    VideoWidget(QWidget *parent = NULL);

    ~VideoWidget();

    void input(tdv::ReadPipe<IplImage*> *framePipe, bool sink);

    void init(tdv::ExceptionReport *report);

    void dispose();

    IplImage* lastFrame();

    void process();

protected:
    virtual void paintEvent(QPaintEvent *event);

private:
    tdv::ReadPipe<IplImage*> *m_framePipe;
    tdv::ProcessRunner *m_procRunner;
    
    IplImage *m_lastFrame, *m_pixmap;
    bool m_end, m_sink;

    QMutex m_imageMutex;
};

#endif /* TDV_VIDEOWIDGET_HPP */
