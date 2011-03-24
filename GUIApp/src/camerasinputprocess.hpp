#ifndef TDV_CAMERASINPUTPROCESS_HPP
#define TDV_CAMERASINPUTPROCESS_HPP

#include <tdvision/captureproc.hpp>
#include <tdvision/processrunner.hpp>
#include <tdvision/exceptionreport.hpp>
#include "inputprocess.hpp"

class CamerasInputProcess: public InputProcess
{
public:
    CamerasInputProcess();
    
    void init(tdv::ExceptionReport *report);
    
    void dispose();    
    
    tdv::ReadPipe<IplImage*>* leftImage()
    {
        return m_capture0.output();
    }

    tdv::ReadPipe<IplImage*>* rightImage()
    {
        return m_capture1.output();
    }
    
    class Factory: public InputProcessFactory
    {
    public:
        InputProcess *create()
        {
            return new CaptureInputProcess;
        }
    private:
    };
private:
    tdv::ProcessRunner *m_procRunner;       
    tdv::CaptureProc m_capture0;
    tdv::CaptureProc m_capture1;    
};

#endif /* TDV_CAMERASINPUTPROCESS_HPP */
