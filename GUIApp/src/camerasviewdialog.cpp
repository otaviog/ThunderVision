#include "appcontext.hpp"
#include "videowidget.hpp"
#include "camerasviewdialog.hpp"

CamerasViewDialog::CamerasViewDialog(AppContext *ctx)
{
    m_appCtx = ctx;    
        
    m_leftVidWid = new VideoWidget(this);
    m_rightVidWid = new VideoWidget(this);                
    
    setupUi(this);
    
    layLeftCam->addWidget(m_leftVidWid);
    layRightCam->addWidget(m_rightVidWid);
}


void CamerasViewDialog::init()
{
    ReadPipeTuple<IplImage*> camOutputs(m_appCtx->enableOutput2());
    
    m_leftVidWid->input(camOutputs.p1, false);
    m_rightVidWid->input(camOutputs.p2, false);
    
    m_leftVidWid->init();
    m_rightVidWid->init();    
}

void CamerasViewDialog::dispose()
{
    m_appCtx->disableOutput2();
    m_leftVidWid->dispose();
    m_rightVidWid->dispose();
}

void CamerasViewDialog::closeEvent(QCloseEvent *event)
{
    
}
