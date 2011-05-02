#include "videowidget.hpp"
#include "disparitydialog.hpp"

DisparityDialog::DisparityDialog(tdv::Reconstruction *rctx,
    QWidget *parent)
    : QDialog(parent)
{
    setupUi(this);
    m_dispView = new VideoWidget(this);
    layDispView->addWidget(m_dispView, 0, 0);
    m_recContext = rctx;
}

void DisparityDialog::init()
{
    tdv::ReadPipe<tdv::FloatImage> *disparityPipe;
    m_recContext->dupDisparityMap(&disparityPipe);
    
    m_dispView->input(disparityPipe);
    m_dispView->init();
}

void DisparityDialog::dispose()
{
    m_recContext->undupDisparityMap();
    m_dispView->dispose();
}

void DisparityDialog::closeEvent(QCloseEvent *event)
{
    emit finished(QDialog::Accepted);
}
