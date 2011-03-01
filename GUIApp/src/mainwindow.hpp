#ifndef TDV_MAINWINDOW_HPP
#define TDV_MAINWINDOW_HPP

#include <QMainWindow>
#include "camerawidget.hpp"

class MainWindow: public QMainWindow
{
public:
    MainWindow();
    
    virtual ~MainWindow();
        
private:
    CameraWidget *m_cameras[2];
};

#endif /* TDV_MAINWINDOW_HPP */
