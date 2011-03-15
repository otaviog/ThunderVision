#ifndef TDV_CAMCALIBRATIONCONTEXT_HPP
#define TDV_CAMCALIBRATIONCONTEXT_HPP

#include <tdvision/processrunner.hpp>
#include <tdvision/calibration.hpp>
#include <tdvision/captureproc.hpp>
#include <tdvision/imagesink.hpp>
#include <tdvision/workunitprocess.hpp>

namespace tdv {
    class ExceptionReport;
}

class CamCalibrationContext
{
public: 
    CamCalibrationContext();
    
    void start(tdv::ExceptionReport *report);

    void stop();
    
    tdv::ReadPipe<IplImage*>* patternDetectProgress(int cam)
    {
        return m_calib.detectionImage();
    }
    
private:
    tdv::ProcessRunner *m_procRunner;       
    tdv::CaptureProc m_capture0;
    tdv::CaptureProc m_capture1;
    tdv::TWorkUnitProcess<tdv::Calibration> m_calib;
    tdv::TWorkUnitProcess<tdv::ImageSink> m_sink0, m_sink1;    
};

#endif /* TDV_CAMCALIBRATIONCONTEXT_HPP */
