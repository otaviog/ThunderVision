#ifndef TDV_MAINWINDOW_HPP
#define TDV_MAINWINDOW_HPP

#include <QMainWindow>
#include "ui_mainwindow.h"
#include "videowidget.hpp"

class MainWindow: public QMainWindow, private Ui::MainWindow
{
    Q_OBJECT;
    
public:
    MainWindow(AppContext *appCtx);
    
    virtual ~MainWindow();        
    
public slots:
    void showCamerasViews();
    
    void showDisparityMap();
        
private:
    AppContext *m_appCtx;
    CamerasViewDialog *m_camsDialog;
    
};

#endif /* TDV_MAINWINDOW_HPP */
