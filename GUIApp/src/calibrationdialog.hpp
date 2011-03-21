#ifndef TDV_CALIBRATIONDIALOG_HPP
#define TDV_CALIBRATIONDIALOG_HPP

#include "errorreport.hpp"
#include "ui_calibrationdialog.h"

class CamCalibrationContext;
class CalibrationWidget;

class CalibrationDialog: public QDialog,
                         protected Ui::CalibrationDialog
{
    Q_OBJECT;
public:
    CalibrationDialog();

    void init();

    void dispose();

    void showCalibration();

    void errorOcurred(const std::exception &ex);

    virtual void closeEvent(QCloseEvent *ev);

public slots:
    void informCriticalError(QString message);
    
    void save();
    
private:
    CalibrationWidget *m_calibWid;
    CamCalibrationContext *m_calibCtx;
    ErrorReport m_errHandle;
};

#endif /* TDV_CALIBRATIONDIALOG_HPP */
