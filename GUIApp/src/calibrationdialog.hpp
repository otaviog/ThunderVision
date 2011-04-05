#ifndef TDV_CALIBRATIONDIALOG_HPP
#define TDV_CALIBRATIONDIALOG_HPP

#include <tdvision/calibration.hpp>
#include "errorreport.hpp"
#include "ui_calibrationdialog.h"

class CalibrationWidget;

class CalibrationDialog: public QDialog,
                         protected Ui::CalibrationDialog
{
    Q_OBJECT;
    
public:
    CalibrationDialog(tdv::Calibration *calib, QWidget *parent = NULL);

    void init();

    void dispose();

protected:
    virtual void closeEvent(QCloseEvent *ev);                                            
                                            
private slots:
    void informCriticalError(QString message);
    
    void save();
    
    void printPattern();
    
private:
    tdv::Calibration *m_calib;
    CalibrationWidget *m_calibWid;    
    ErrorReport m_errHandle;
};

#endif /* TDV_CALIBRATIONDIALOG_HPP */
