#include <QMessageBox>
#include <QFileDialog>
#include <tdvision/thunderlang.hpp>
#include <tdvision/writeexception.hpp>
#include "calibrationwidget.hpp"
#include "calibrationdialog.hpp"

CalibrationDialog::CalibrationDialog(tdv::Calibration *calibCtx)
{
    setupUi(this);
    
    assert(calibCtx != NULL);
    
    m_calib = calibCtx;
    
    m_calibWid = new CalibrationWidget;        
    
    lyCalibWid->addWidget(m_calibWid);
    
    connect(&m_errHandle, SIGNAL(informError(QString)),
            this, SLOT(informCriticalError(QString)));
    connect(pbSave, SIGNAL(clicked()),
            this, SLOT(save()));        
}

void CalibrationDialog::init()
{
    m_calibWid->init(m_calib->detectionImage(), true);    
    m_calib->observer(m_calibWid);
}

void CalibrationDialog::dispose()
{
    m_calibWid->dispose();               
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
    QString filename = QFileDialog::getSaveFileName(
        this, tr("Save Calibration"), QString(),
        tr("ThunderLang (*.tl)"));
    
    if ( !filename.isEmpty() )
    {
        try
        {
            tdv::ThunderLangWriter writer;
            tdv::ThunderSpec spec;
            spec.camerasDesc("default") = m_calib->camerasDesc();
            writer.write(filename.toStdString(), spec);
        }
        catch (const tdv::WriteException &ex)
        {
        }
    }        
}
