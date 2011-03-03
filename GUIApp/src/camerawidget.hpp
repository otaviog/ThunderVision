#ifndef TDV_CAMERAWIDGET_HPP
#define TDV_CAMERAWIDGET_HPP

#include <tdvision/pipe.hpp>
#include <tdvision/workunit.hpp>
#include <cv.h>
#include <QWidget>
#include <QThread>

class CameraWidget: public QWidget, public tdv::WorkUnit
{
public:    
    CameraWidget(tdv::ReadPipe<IplImage*> *framePipe);
    
    ~CameraWidget();
    
    void process();
    
protected:
    virtual void paintEvent(QPaintEvent *event);
    
private:            
    tdv::ReadPipe<IplImage*> *m_framePipe;
    IplImage *m_lastFrame;    
    bool m_end;
};

#endif /* TDV_CAMERAWIDGET_HPP */
