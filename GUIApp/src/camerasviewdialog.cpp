#include "videowidget.hpp"
#include "camerasviewdialog.hpp"

CamerasViewDialog::CamerasViewDialog(tdv::TDVContext *ctx)
{
    m_ctx = ctx;    
        
    m_leftVidWid = new VideoWidget(this);
    m_rightVidWid = new VideoWidget(this);                
    
    setupUi(this);
    
    layLeftCam->addWidget(m_leftVidWid);
    layRightCam->addWidget(m_rightVidWid);
}


void CamerasViewDialog::init()
{
    tdv::ReadPipe<IplImage*> *lpipe, *rpipe;
    
    m_ctx->dupInputSource(&lpipe, &rpipe);
    
    m_leftVidWid->input(lpipe, false);
    m_rightVidWid->input(rpipe, false);
    
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
