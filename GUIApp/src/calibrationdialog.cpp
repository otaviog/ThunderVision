#include "camcalibrationcontext.hpp"
#include "calibrationwidget.hpp"
#include "calibrationdialog.hpp"

CalibrationDialog::CalibrationDialog()
{
    setupUi(this);
    m_calibWid = new CalibrationWidget;        
    lyCalibWid->addWidget(m_calibWid);
}

void CalibrationDialog::init()
{
    m_calibCtx = new CamCalibrationContext(10);
    m_calibCtx->init(this);    
    m_calibWid->init(m_calibCtx->patternDetectProgress(), true);    
}

void CalibrationDialog::dispose()
{
    if ( m_calibCtx != NULL )
    {
        m_calibCtx->dispose();
        m_calibWid->dispose();
        delete m_calibCtx;
        m_calibCtx = NULL;
    }        
}

void CalibrationDialog::errorOcurred(const std::exception &ex)
{
}

void CalibrationDialog::closeEvent(QCloseEvent *ev)
{
    dispose();
}

