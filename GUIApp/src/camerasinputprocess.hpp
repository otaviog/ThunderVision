#ifndef TDV_CAMERASINPUTPROCESS_HPP
#define TDV_CAMERASINPUTPROCESS_HPP

#include <tdvision/captureproc.hpp>
#include <tdvision/processrunner.hpp>
#include "inputprocess.hpp"

class CamerasInputProcess: public InputProcess
{
public:
    CamerasInputProcess();
    
    void init(tdv::ExceptionReport *report);
    
    void dispose();    
    
    tdv::ReadPipe<IplImage*>* leftImgOutput()
    {
        return m_capture0.output();
    }

    tdv::ReadPipe<IplImage*>* rightImgOutput()
    {
        return m_capture1.output();
    }
    
    class Factory: public InputProcessFactory
    {
    public:
        InputProcess *create()
        {
            return new CamerasInputProcess;
        }
    private:
    };
    
private:
    tdv::ProcessRunner *m_procRunner;       
    tdv::CaptureProc m_capture0;
    tdv::CaptureProc m_capture1;    
};

#endif /* TDV_CAMERASINPUTPROCESS_HPP */
