#ifndef TDV_CAMERASVIEWDIALOG_HPP
#define TDV_CAMERASVIEWDIALOG_HPP

#include <tdvision/tdvcontext.hpp>
#include "ui_camerasviewdialog.h"

class VideoWidget;

class CamerasViewDialog: public QDialog, Ui::CamerasView
{
    Q_OBJECT;
public:
    CamerasViewDialog(tdv::TDVContext *ctx);
    
    void init();    

private slots:
    void showCalibrationDlg();
    
protected:
    void closeEvent(QCloseEvent *event);
    
    void dispose();
private:
    tdv::TDVContext *m_ctx;
    VideoWidget *m_leftVidWid;
    VideoWidget *m_rightVidWid;
};

#endif /* TDV_CAMERASVIEWDIALOG_HPP */
