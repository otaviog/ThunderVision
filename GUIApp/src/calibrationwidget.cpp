#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QMessageBox>
#include <tdvision/captureproc.hpp>
#include <tdvision/imagesink.hpp>
#include <tdvision/calibration.hpp>
#include <tdvision/rgbconv.hpp>
#include "camerawidget.hpp"
#include "camerasviewwidget.hpp"
#include "calibrationwidget.hpp"

CalibrationWidget::CalibrationWidget()
    : m_capture0(0), m_capture1(1)
{
    m_procRunner = NULL;
    
    QHBoxLayout *hbox = new QHBoxLayout;    
    QVBoxLayout *vbox = new QVBoxLayout;
        
    QPushButton *pbOpenCameras = new QPushButton("Start Calibration"),
        *pbCloseCameras = new QPushButton("Stop Calibration");
        
    m_camWid = new CameraWidget(m_calib.detectionImage(), true);
    
    connect(pbOpenCameras, SIGNAL(clicked()),
            this, SLOT(openCameras()));
    
    connect(pbCloseCameras, SIGNAL(clicked()),
            this, SLOT(closeCameras()));
    
    connect(this, SIGNAL(destroyed()),
            this, SLOT(closeCameras()));

    hbox->addWidget(pbOpenCameras);
    hbox->addWidget(pbCloseCameras);
    vbox->addLayout(hbox);
    vbox->addWidget(m_camWid);        
    
    setLayout(vbox);
}

void CalibrationWidget::openCameras()
{    
    if ( m_procRunner == NULL )
    {                
        m_calib.input(m_capture0.colorImage(),
                      m_capture1.colorImage());
                
        m_sink0.input(m_capture0.output());
        m_sink1.input(m_capture1.output());
                            
        tdv::Process *procs[] = {
            &m_capture0, &m_capture1, 
            m_camWid, &m_calib,
            &m_sink0, &m_sink1, NULL };
        
        m_procRunner = new tdv::ProcessRunner(procs, this);     
        m_procRunner->run();
    }
}

void CalibrationWidget::closeCameras()
{
    if ( m_procRunner != NULL )
    {
        m_capture0.finish();
        m_capture1.finish();
    
        m_procRunner->join();
        delete m_procRunner;
        
        m_procRunner = NULL;
    }
    
}

void CalibrationWidget::errorOcurred(const std::exception &err)
{
    QMessageBox::critical(this, tr(""), err.what());
}
