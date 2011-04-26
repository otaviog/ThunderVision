#include <iostream>
#include "videowidget.hpp"
#include "calibrationdialog.hpp"
#include "camerasviewdialog.hpp"

CamerasViewDialog::CamerasViewDialog(tdv::TDVContext *ctx, QWidget *parent)
    : QDialog(parent)
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
    emit finished(QDialog::Accepted);
}

void CamerasViewDialog::showCalibrationDlg()
{
    tdv::Calibration *calib = m_ctx->runCalibration();
    
    if ( calib != NULL )
    {
        m_calibDlg = new CalibrationDialog(calib);
        m_calibDlg->init();
        m_calibDlg->show();
        pbCalibrate->setEnabled(false);
        connect(m_calibDlg, SIGNAL(finished(int)),
                this, SLOT(doneCalibrationDlg()));
    }
}

void CamerasViewDialog::doneCalibrationDlg()
{
    if ( m_calibDlg != NULL )
    {
        m_calibDlg->dispose();
        m_calibDlg = NULL;
        pbCalibrate->setEnabled(true);
    }
}
