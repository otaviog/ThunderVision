#include <QMessageBox>
#include <QFileDialog>
#include <QPrinter>
#include <QPrintDialog>
#include <QPainter>
#include <tdvision/thunderlang.hpp>
#include <tdvision/writeexception.hpp>
#include "videowidget.hpp"
#include "calibrationdialog.hpp"

CalibrationDialog::CalibrationDialog(tdv::Calibration *calibCtx, 
                                     QWidget *parent)
    : QDialog(parent)
{
    setupUi(this);

    assert(calibCtx != NULL);

    m_calib = calibCtx;
    
    m_calib->stepMode();
    
    m_videoWid = new VideoWidget;    
    lyCamWid->addWidget(m_videoWid);

    connect(&m_errHandle, SIGNAL(informError(QString)),
            this, SLOT(informCriticalError(QString)));
    connect(pbSave, SIGNAL(clicked()),
            this, SLOT(save()));    
    connect(pbPrintPattern, SIGNAL(clicked()),
            this, SLOT(printPattern()));
    connect(pbPassFrame, SIGNAL(clicked()),
            this, SLOT(passFrame()));
}

void CalibrationDialog::init()
{
    m_videoWid->input(m_calib->detectionImage());
    m_videoWid->init();
    m_calib->observer(this);
}

void CalibrationDialog::dispose()
{
    m_videoWid->dispose();
}

void CalibrationDialog::closeEvent(QCloseEvent *ev)
{
    emit finished(QDialog::Accepted);
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

void CalibrationDialog::printPattern()
{
    QPrinter printer;

    QPrintDialog *dialog = new QPrintDialog(&printer, this);
    dialog->setWindowTitle(tr("Print Document"));
    
    if (dialog->exec() != QDialog::Accepted)
        return;
  
    QPainter painter;
    painter.begin(&printer);

    QImage img(QString(":/imgs/chessboard.png"));
    
    painter.drawImage(QPoint(0, 0), img);
    
    painter.end();
}

void CalibrationDialog::passFrame()
{
    m_calib->step();
}

void CalibrationDialog::calibrationUpdate(const tdv::Calibration &calib)
{
    float percent = float(calib.framesProcessed())/float(calib.numFrames());
    QMetaObject::invokeMethod(pbProgress, "setValue", Qt::QueuedConnection,
                              Q_ARG(int, percent*100));
    
    if ( calib.framesProcessed() == calib.numFrames() )
    {
        QMetaObject::invokeMethod(lbStatus, "setText", Qt::QueuedConnection,
                                  Q_ARG(QString, tr("Calibration done")));

    }
    else
    {
        QMetaObject::invokeMethod(lbStatus, "setText", Qt::QueuedConnection,
                                  Q_ARG(QString, tr("Calibrating...")));
    }
}

void CalibrationDialog::passFrame()
{
    m_calib->stepCalibration();
}
