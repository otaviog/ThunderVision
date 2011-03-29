#ifndef TDV_CALIBRATIONWIDGET_HPP
#define TDV_CALIBRATIONWIDGET_HPP

#include <QWidget>
#include <QLabel>
#include <QProgressBar>
#include <tdvision/exceptionreport.hpp>
#include "ui_calibrationwidget.h"
#include "calibrationcontext.hpp"

class VideoWidget;

class CalibrationWidget: 
    public QWidget, 
    protected Ui::CalibrationWidget,
    public tdv::CalibrationObserver
{
    Q_OBJECT;
    
public:
    CalibrationWidget();
    
    void init(tdv::ReadPipe<IplImage*> *patternDetect, bool sink);
    
    void dispose();    
    
    virtual void calibrationUpdate(const tdv::Calibration &calib);
                                                    
private:
    VideoWidget *m_videoWid;
    QLabel *m_lbStatus;
    QProgressBar *m_pbProgress;
};

#endif /* TDV_CALIBRATION_HPP */
