#ifndef TDV_CAMERASVIEWWIDGET_HPP
#define TDV_CAMERASVIEWWIDGET_HPP

#include <QWidget>
#include <tdvision/pipe.hpp>
#include <cv.h>
#include <tdvision/processrunner.hpp>

class CameraWidget;

class CamerasViewWidget: public QWidget, tdv::ProcessExceptionReport
{
    Q_OBJECT ;
public:    
    CamerasViewWidget(tdv::ReadPipe<IplImage*> *leftPipe, tdv::ReadPipe<IplImage*> *rightPipe);
        
    virtual ~CamerasViewWidget()
    { }
    
    void start();
 
    void stop();
    
    void errorOcurred(const std::exception &err);

public slots:
    void photoChar();
    
private:
    CameraWidget *leftCamW, *rightCamW;
};

#endif /* TDV_CAMERASVIEWWIDGET_HPP */
