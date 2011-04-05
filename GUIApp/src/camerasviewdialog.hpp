#ifndef TDV_CAMERASVIEWDIALOG_HPP
#define TDV_CAMERASVIEWDIALOG_HPP

#include <tdvision/tdvcontext.hpp>
#include "ui_camerasviewdialog.h"

class VideoWidget;
class CalibrationDialog;

class CamerasViewDialog: public QDialog, Ui::CamerasView
{
    Q_OBJECT;
public:
    CamerasViewDialog(tdv::TDVContext *ctx, QWidget *parent = NULL);
    
    void init();    

    void dispose();
                  
private slots:
    void showCalibrationDlg();
    
    void doneCalibrationDlg();
        
protected:
    void closeEvent(QCloseEvent *event);        
    
private:
    tdv::TDVContext *m_ctx;
    VideoWidget *m_leftVidWid;
    VideoWidget *m_rightVidWid;
    
    CalibrationDialog *m_calibDlg;
};

#endif /* TDV_CAMERASVIEWDIALOG_HPP */
