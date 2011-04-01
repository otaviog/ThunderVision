#include "videowidget.hpp"
#include "calibrationdialog.hpp"
#include "camerasviewdialog.hpp"

CamerasViewDialog::CamerasViewDialog(tdv::TDVContext *ctx)
{
    m_ctx = ctx;    
        
    m_leftVidWid = new VideoWidget(this);
    m_rightVidWid = new VideoWidget(this);                
    
    setupUi(this);
    
    layLeftCam->addWidget(m_leftVidWid, 0, 0);
    layRightCam->addWidget(m_rightVidWid, 0, 0);
    
    connect(pbCalibrate, SIGNAL(clicked()),
            this, SLOT(showCalibrationDlg()));
}

void CamerasViewDialog::init()
{
    tdv::ReadPipe<CvMat*> *lpipe, *rpipe;
    
    m_ctx->dupInputSource(&lpipe, &rpipe);
    
    m_leftVidWid->input(lpipe);
    m_rightVidWid->input(rpipe);
    
    m_leftVidWid->init();
    m_rightVidWid->init();    
}

void CamerasViewDialog::dispose()
{
    m_ctx->undupInputSource();
    m_leftVidWid->dispose();
    m_rightVidWid->dispose();
}

void CamerasViewDialog::closeEvent(QCloseEvent *event)
{
    dispose();
}

void CamerasViewDialog::showCalibrationDlg()
{
    
}
