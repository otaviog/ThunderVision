#ifndef TDV_CAMCALIBRATIONCONTEXT_HPP
#define TDV_CAMCALIBRATIONCONTEXT_HPP

#include <tdvision/processrunner.hpp>
#include <tdvision/calibration.hpp>
#include <tdvision/captureproc.hpp>
#include <tdvision/imagesink.hpp>
#include <tdvision/workunitprocess.hpp>

class CamCalibrationContext
{
public: 
    CamCalibrationContext();
    
    void start();

    void stop();
    
    tdv::ReadPipe<IplImage*>* patternDetectProgress(int cam)
    {
    }
    
private:
    tdv::ProcessRunner *m_procRunner;       
    tdv::CaptureProc m_capture0;
    tdv::CaptureProc m_capture1;
    tdv::TWorkUnitProcess<tdv::Calibration> m_calib;
    tdv::TWorkUnitProcess<tdv::ImageSink> m_sink0, m_sink1;
    
    CameraWidget *m_camWid;
};

#endif /* TDV_CAMCALIBRATIONCONTEXT_HPP */
