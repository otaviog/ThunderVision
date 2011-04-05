#ifndef TDV_MAINWINDOW_HPP
#define TDV_MAINWINDOW_HPP

#include <QMainWindow>
#include <tdvision/tdvcontext.hpp>
#include <tdvision/reconstruction.hpp>
#include <tdvision/stereoinputsource.hpp>
#include "ui_mainwindow.h"
#include "videowidget.hpp"

class CamerasViewDialog;
class RectificationViewDialog;

class MainWindow: public QMainWindow, private Ui::MainWindow
{
    Q_OBJECT;
    
public:
    MainWindow(tdv::TDVContext *ctx);
    
    virtual ~MainWindow();        
                         
    void start(tdv::StereoInputSource *inputSrc);
                
public slots:
    void showCamerasViews();
    
    void showDisparityMap();
    
    void showReconstructionConfig();

    void showRectification();

    void playReconstruction();

    void stepReconstruction();
      
    void pauseReconstruction();
    
protected:
    void closeEvent(QCloseEvent *event);
    
    void dispose();
    
    void initReconstruction();              

private slots:
    
    void doneCamerasViews();
    
private:
    tdv::TDVContext *m_ctx;
    tdv::Reconstruction *m_reconst;
    
    CamerasViewDialog *m_camsDialog;
    RectificationViewDialog *m_rectDialog;
};

#endif /* TDV_MAINWINDOW_HPP */
