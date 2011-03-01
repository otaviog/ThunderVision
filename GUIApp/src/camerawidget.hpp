#ifndef TDV_CAMERAWIDGET_HPP
#define TDV_CAMERAWIDGET_HPP

#include <tdvision/pipe.hpp>
#include <cv.h>
#include <QWidget>
#include <QThread>

class CameraWidget: public QWidget
{
public:    
    CameraWidget(tdv::ReadPipe<IplImage*> *framePipe);
    
    ~CameraWidget();
    
    void init();
    
protected:
    virtual void paintEvent(QPaintEvent *event);
    
private:        
    struct PipeWatch: public QThread
    {
    public:
        PipeWatch(CameraWidget *self)
            : QThread(self), m_self(self)
        {
        }
        
        void run();
        
    private:
        CameraWidget *m_self;
    };
    
    tdv::ReadPipe<IplImage*> *m_framePipe;
    IplImage *m_lastFrame;
    PipeWatch m_watcher;
};

#endif /* TDV_CAMERAWIDGET_HPP */
