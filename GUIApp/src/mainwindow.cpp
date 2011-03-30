#include "camerasviewdialog.hpp"
#include "mainwindow.hpp"

MainWindow::MainWindow(tdv::TDVContext *ctx)
{
    m_camsDialog = NULL;
    m_ctx = ctx;
    m_reconst = NULL;
    
    setupUi(this);
    connect(pbCamerasView, SIGNAL(clicked()),
            this, SLOT(showCamerasViews()));
}

MainWindow::~MainWindow()
{
    
}

void MainWindow::playReconstruction()
{
    if ( m_reconst != NULL )
    {
        m_reconst->continuous();
    }
}

void MainWindow::stepReconstruction()
{
    if ( m_reconst != NULL )
    {
        m_reconst->step();
    }
}

void MainWindow::pauseReconstruction()
{
    if ( m_reconst != NULL )
    {
        m_reconst->pause();
    }
}

void MainWindow::showCamerasViews()
{
    if ( m_camsDialog == NULL )
    {
        m_camsDialog = new CamerasViewDialog(m_ctx);
        m_camsDialog->init();
        m_camsDialog->show();        
    }
}
    
void MainWindow::showDisparityMap()
{
}

void MainWindow::showReconstructionConfig()
{
}

void MainWindow::showRectification()
{
}
