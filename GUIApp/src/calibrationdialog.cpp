#include <QMessageBox>
#include "camcalibrationcontext.hpp"
#include "calibrationwidget.hpp"
#include "calibrationdialog.hpp"

CalibrationDialog::CalibrationDialog()
{
    setupUi(this);
    m_calibWid = new CalibrationWidget;        
    lyCalibWid->addWidget(m_calibWid);
    
    connect(&m_errHandle, SIGNAL(informError(QString)),
            this, SLOT(informCriticalError(QString)));
}

void CalibrationDialog::init()
{
    m_calibCtx = new CamCalibrationContext(10);
    m_calibCtx->init(&m_errHandle);    
    m_calibWid->init(m_calibCtx->patternDetectProgress(), true);    
    m_calibCtx->calibObserver(m_calibWid);
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

void CalibrationDialog::closeEvent(QCloseEvent *ev)
{
    dispose();
}

void CalibrationDialog::informCriticalError(QString message)
{
    QMessageBox::critical(this, "Error", message);
}
