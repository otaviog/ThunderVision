#ifndef TDV_MAINWINDOW_HPP
#define TDV_MAINWINDOW_HPP

#include <QMainWindow>
#include <tdvision/tdvcontext.hpp>
#include <tdvision/reconstruction.hpp>
#include <tdvision/stereoinputsource.hpp>
#include <tdvision/dilate.hpp>
#include "ui_mainwindow.h"
#include "videowidget.hpp"
#include "reprojectionview.hpp"

class CamerasViewDialog;
class RectificationViewDialog;
class DisparityDialog;

class MainWindow: public QMainWindow, private Ui::MainWindow
{
    Q_OBJECT;
    
public:
    MainWindow(tdv::TDVContext *ctx);
    
    virtual ~MainWindow();        
                         
    void start(tdv::StereoInputSource *inputSrc);
                
public slots:
    void showCamerasViews();        

    void showRectification();
    
    void showDisparity();
    
    void playReconstruction();

    void stepReconstruction();
      
    void pauseReconstruction();
    
protected:
    void closeEvent(QCloseEvent *event);
    
    void dispose();
    
    void initReconstruction();              

private slots:    
    void doneCamerasViews();
    
    void doneRectification();
    
    void doneDisparity();    
    
private:
    tdv::TDVContext *m_ctx;
    tdv::Reconstruction *m_reconst;
    
    CamerasViewDialog *m_camsDialog;
    RectificationViewDialog *m_rectDialog;
    DisparityDialog *m_dispDialog;
    ReprojectionView *m_reprView;
};

#endif /* TDV_MAINWINDOW_HPP */
