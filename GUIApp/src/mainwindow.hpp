#ifndef TDV_MAINWINDOW_HPP
#define TDV_MAINWINDOW_HPP

#include <QMainWindow>
#include <tdvision/tdvcontext.hpp>
#include <tdvision/reconstruction.hpp>
#include "ui_mainwindow.h"
#include "camerasviewdialog.hpp"
#include "videowidget.hpp"

class MainWindow: public QMainWindow, private Ui::MainWindow
{
    Q_OBJECT;
    
public:
    MainWindow(tdv::TDVContext *ctx);
    
    virtual ~MainWindow();        
    
public slots:
    void showCamerasViews();
    
    void showDisparityMap();
    
    void showReconstructionConfig();

    void showRectification();

    void playReconstruction();

    void stepReconstruction();
      
    void pauseReconstruction();
    
private:
    tdv::TDVContext *m_ctx;
    tdv::Reconstruction *m_reconst;
    CamerasViewDialog *m_camsDialog;    
};

#endif /* TDV_MAINWINDOW_HPP */
