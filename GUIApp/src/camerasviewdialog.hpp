#ifndef TDV_CAMERASVIEWDIALOG_HPP
#define TDV_CAMERASVIEWDIALOG_HPP

#include "ui_camerasviewdialog.h"

class AppContext;
class VideoWidget;

class CamerasViewDialog: public QDialog, Ui::CamerasView
{
public:
    CamerasViewDialog(AppContext *appCtx);
    
    void inti();
    
protected:
    void closeEvent(QCloseEvent *event);
    
private:
    AppContext *m_appCtx;
    VideoWidget *m_leftVidWid;
    VideoWidget *m_rightVidWid;
};

#endif /* TDV_CAMERASVIEWDIALOG_HPP */
