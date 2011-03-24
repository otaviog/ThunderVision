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

class CalibrationContext
{
public: 
    CalibrationContext(size_t numFrames);
    
    void init(tdv::ReadPipe<IplImage*> *leftImgInput,
              tdv::ReadPipe<IplImage*> *rightImgInput,
              tdv::ExceptionReport *report);

    void dispose();
    
    void calibObserver(tdv::CalibrationObserver *obs)
    {
        m_calib.observer(obs);
    }
    
    tdv::ReadPipe<IplImage*>* patternDetectProgress()
    {
        return m_calib.detectionImage();
    }
    
    const tdv::CamerasDesc& camerasDesc() const
    {
        return m_calib.camerasDesc();
    }
    
private:
    tdv::ProcessRunner *m_procRunner;       
    tdv::TWorkUnitProcess<tdv::ImageSink> m_sink0, m_sink1;    
    tdv::Calibration m_calib;
    tdv::WorkUnitProcess *m_calibProc;
};

#endif /* TDV_CAMCALIBRATIONCONTEXT_HPP */
