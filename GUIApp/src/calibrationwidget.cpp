#include <iostream>
#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QMessageBox>
#include <tdvision/captureproc.hpp>
#include <tdvision/imagesink.hpp>
#include <tdvision/calibration.hpp>
#include <tdvision/rgbconv.hpp>
#include "videowidget.hpp"
#include "camerasviewwidget.hpp"
#include "calibrationwidget.hpp"

CalibrationWidget::CalibrationWidget()
{    
    setupUi(this);
    
    m_videoWid = new VideoWidget;    
    lyCamWid->addWidget(m_videoWid);
}

void CalibrationWidget::init(tdv::ReadPipe<IplImage*> *patternDetect, bool sink)
{
    m_videoWid->input(patternDetect, sink);
    m_videoWid->init();
}

void CalibrationWidget::dispose()
{
    m_videoWid->dispose();
}

void CalibrationWidget::calibrationUpdate(const tdv::Calibration &calib)
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
