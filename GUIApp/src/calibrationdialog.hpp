#ifndef TDV_CALIBRATIONDIALOG_HPP
#define TDV_CALIBRATIONDIALOG_HPP

#include <tdvision/calibration.hpp>
#include "errorreport.hpp"
#include "ui_calibrationdialog.h"

class VideoWidget;

class CalibrationDialog: public QDialog,
                         public tdv::CalibrationObserver,
                         protected Ui::CalibrationDialog
{
    Q_OBJECT;
    
public:
    CalibrationDialog(tdv::Calibration *calib, QWidget *parent = NULL);

    void init();

    void dispose();

    void calibrationUpdate(const tdv::Calibration &calib);
    
protected:
    virtual void closeEvent(QCloseEvent *ev);                                            
                                            
private slots:
    void informCriticalError(QString message);
    
    void save();
    
    void printPattern();
    
private:
    tdv::Calibration *m_calib;
    VideoWidget *m_videoWid;
    ErrorReport m_errHandle;
};

#endif /* TDV_CALIBRATIONDIALOG_HPP */
