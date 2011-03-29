#ifndef TDV_CAMERASTEREOINPUTSOURCE_HPP
#define TDV_CAMERASTEREOINPUTSOURCE_HPP

#include <tdvbasic/common.hpp>
#include "process.hpp"
#include "processgroup.hpp"
#include "stereoinputsource.hpp"
#include "captureproc.hpp"

TDV_NAMESPACE_BEGIN

class CaptureStereoInputSource: public StereoInputSource, public Process
{
public:
    CaptureStereoInputSource();
    
    void init();

    void init(const std::string &filename1, const std::string &filename2);    
        
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
    
    void process();
    
    void finish()
    {
        m_stopCap = true;
    }
    
private:
    Capture m_capture1;
    Capture m_capture2;
    bool m_stopCap;
    Process *m_procs[3];
};

TDV_NAMESPACE_END

#endif /* TDV_CAMERASTEREOINPUTSOURCE_HPP */
