#include <QMessageBox>
#include <QFileDialog>
#include <tdvision/thunderlang.hpp>
#include <tdvision/writeexception.hpp>
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
    connect(pbSave, SIGNAL(clicked()),
            this, SLOT(save()));
    
    //pbSabe->setEnable(false);
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

void CalibrationDialog::save()
{
    if ( m_calibCtx == NULL )
        return ;
    
    QString filename = QFileDialog::getSaveFileName(
        this, tr("Save Calibration"), QString(),
        tr("ThunderLang (*.tl)"));
    
    if ( !filename.isEmpty() )
    {
        try
        {
            tdv::ThunderLangWriter writer;
            tdv::ThunderSpec spec;
            spec.camerasDesc("default") = m_calibCtx->camerasDesc();
            writer.write(filename.toStdString(), spec);
        }
        catch (const tdv::WriteException &ex)
        {
        }
    }        
}
