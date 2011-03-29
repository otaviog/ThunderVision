#include "camerasviewdialog.hpp"
#include "mainwindow.hpp"

MainWindow::MainWindow(TDVContext *ctx)
{
    m_camsDialog = NULL;
    m_ctx = ctx;
    setupUi(this);
    connect(pbCamerasView, SIGNAL(clicked()),
            this, SLOT(showCamerasView));
}

MainWindow::~MainWindow()
{
    
}

void MainWindow::startReconstruction()
{
    if ( m_stereoMatcher != NULL )
    {
        m_stereoMatcher = m_appCtx->stereoMatcher();
    }
}

void MainWindow::playReconstruction()
{
    m_appCtx->play();
}

void MainWindow::stepReconstruction()
{
    m_appCtx->step();
}

void MainWindow::pauseReconstruction()
{
    m_appCtx->pause();
}

void MainWindow::showCamerasViews()
{
    if ( m_camsDialog == NULL )
    {
        m_camsDialog = new CamerasViewDialog(m_ctx);
        m_camsDialog->show();
        pbCamerasView->setEnabled(false);
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
