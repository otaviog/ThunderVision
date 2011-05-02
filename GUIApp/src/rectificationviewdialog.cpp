#include "rectificationviewdialog.hpp"

RectificationViewDialog::RectificationViewDialog(tdv::Reconstruction *rctx,
    QWidget *parent)
    : QDialog(parent)
{
    setupUi(this);
    
    m_rctx = rctx;
    m_leftVidWid = new VideoWidget(this);
    m_rightVidWid = new VideoWidget(this);
    
    layVid0->addWidget(m_leftVidWid, 0, 0);
    layVid1->addWidget(m_rightVidWid, 0, 0);
}

void RectificationViewDialog::init()
{
    tdv::ReadPipe<tdv::FloatImage> *lpipe, *rpipe;
    
    m_rctx->dupRectification(&lpipe, &rpipe);
    
    m_leftVidWid->input(lpipe);
    m_rightVidWid->input(rpipe);
    m_leftVidWid->init();
    m_rightVidWid->init();    
}

void RectificationViewDialog::closeEvent(QCloseEvent *event)
{
    emit finished(QDialog::Accepted);
}
    
void RectificationViewDialog::dispose()
{
    if ( m_rctx != NULL )
    {
        m_rctx->undupRectification();
    
        m_leftVidWid->dispose();
        m_rightVidWid->dispose();
    }
}
