#include <QProgressDialog>
#include <QMessageBox>
#include "camerasviewdialog.hpp"
#include "selectinputdialog.hpp"
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

void MainWindow::start(tdv::StereoInputSource *inputSrc)
{
    if ( inputSrc == NULL )
    {
        SelectInputDialog inputDlg;
        if ( inputDlg.exec() == QDialog::Accepted )
        {
            QProgressDialog loadDlg("Loading", "", 0, 0, 
                                    NULL, Qt::FramelessWindowHint);
            loadDlg.show();
            try
            {
                
                inputSrc = inputDlg.createInputSource();
            }
            catch (const tdv::Exception &ex)
            {
                QMessageBox::critical(this, tr("Error while creating input sources"),
                                      tr(ex.what()));
            }
            
            loadDlg.close();
        }
    }
    
    if ( inputSrc != NULL )
    {
        m_ctx->start(inputSrc);
    }
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
        connect(m_camsDialog, SIGNAL(finished(int)),
                this, SLOT(camerasViewsDone()));
        pbCamerasView->setEnabled(false);
    }
}

void MainWindow::camerasViewsDone()
{
    if( m_camsDialog != NULL )
    {
        m_camsDialog = NULL;
        pbCamerasView->setEnabled(true);
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
