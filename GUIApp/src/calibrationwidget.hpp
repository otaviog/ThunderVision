#ifndef TDV_CALIBRATIONWIDGET_HPP
#define TDV_CALIBRATIONWIDGET_HPP

#include <QWidget>
#include <QLabel>
#include <QProgressBar>
#include <tdvision/exceptionreport.hpp>
#include "ui_calibrationwidget.h"
#include "camcalibrationcontext.hpp"

class CameraWidget;

class CalibrationWidget: 
    public QWidget, 
    public tdv::ExceptionReport, 
    protected Ui::CalibrationWidget,
    public tdv::CalibrationObserver
{
    Q_OBJECT;
    
public:
    CalibrationWidget();
    
    void init(tdv::ReadPipe<IplImage*> *patternDetect, bool sink);
    
    void dispose();
    
    void errorOcurred(const std::exception &err);
    
    virtual void calibrationUpdate(const tdv::Calibration &calib);
                                                    
private:
    CameraWidget *m_camWid;
    QLabel *m_lbStatus;
    QProgressBar *m_pbProgress;
};

#endif /* TDV_CALIBRATION_HPP */
