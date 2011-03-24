#include "mainwindow.hpp"

MainWindow::MainWindow(AppContext *appCtx)
{
    m_camsDialog = NULL;
    
    setupUi(this);
    connect(pbCamerasView, SIGNAL(clicked()),
            this, SLOT(showCamerasView));
}

MainWindow::~MainWindow()
{
    
}

void MainWindow::showCamerasViews()
{
    if ( m_camsDialog == NULL )
    {
        m_camsDialog = new CamerasViewDialog(appCtx);
        m_camsDialog->show();
        pbCamerasView->setEnabled(false);
    }
}
    
void MainWindow::showDisparityMap()
{
}
