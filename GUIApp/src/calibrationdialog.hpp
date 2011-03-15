#ifndef TDV_CALIBRATIONDIALOG_HPP
#define TDV_CALIBRATIONDIALOG_HPP

#include <tdvision/exceptionreport.hpp>
#include "ui_calibrationdialog.h"

class CamCalibrationContext;
class CalibrationWidget;

class CalibrationDialog: public QDialog, 
                         protected Ui::CalibrationDialog,
                         public tdv::ExceptionReport
{
public:
    CalibrationDialog();
    
    void init();
        
    void dispose();
    
    void showCalibration();
    
    void errorOcurred(const std::exception &ex);
    
    virtual void closeEvent(QCloseEvent *ev);
    
private:
    CalibrationWidget *m_calibWid;
    CamCalibrationContext *m_calibCtx;
};

#endif /* TDV_CALIBRATIONDIALOG_HPP */
