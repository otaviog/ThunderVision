#ifndef TDV_CAMERASTEREOINPUTSOURCE_HPP
#define TDV_CAMERASTEREOINPUTSOURCE_HPP

#include <tdvbasic/common.hpp>
#include "processgroup.hpp"
#include "stereoinputsource.hpp"
#include "captureproc.hpp"

TDV_NAMESPACE_BEGIN

class CameraStereoInputSource: public StereoInputSource
{
public:
    CameraStereoInputSource();
    
    ReadPipe<IplImage*> *leftImgOutput()
    {
        return m_capture1.output();
    }
    
    ReadPipe<IplImage*> *rightImgOutput()
    {
        return m_capture2.output();
    }
    
    Process** processes()
    {
        return m_procs;
    }
    
private:
    CaptureProc m_capture1;
    CaptureProc m_capture2;
    Process *m_procs[3];
};

TDV_NAMESPACE_END

#endif /* TDV_CAMERASTEREOINPUTSOURCE_HPP */
