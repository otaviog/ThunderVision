#ifndef TDV_CALIBRATIONWIDGET_HPP
#define TDV_CALIBRATIONWIDGET_HPP

#include <QWidget>
#include <tdvision/processrunner.hpp>
#include <tdvision/calibration.hpp>
#include <tdvision/captureproc.hpp>
#include <tdvision/imagesink.hpp>
#include <tdvision/workunitprocess.hpp>

class CameraWidget;

class CalibrationWidget: public QWidget, public tdv::ProcessExceptionReport
{
    Q_OBJECT;
public:
    CalibrationWidget();
    
    void errorOcurred(const std::exception &err);
                                                
public slots:
    void openCameras();
    
    void closeCameras();
            
private:
    tdv::ProcessRunner *m_procRunner;       
    tdv::CaptureProc m_capture0;
    tdv::CaptureProc m_capture1;
    tdv::TWorkUnitProcess<tdv::Calibration> m_calib;
    tdv::TWorkUnitProcess<tdv::ImageSink> m_sink0, m_sink1;
    
    CameraWidget *m_camWid;

};

#endif /* TDV_CALIBRATION_HPP */
