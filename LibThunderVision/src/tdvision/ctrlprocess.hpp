#ifndef TDV_CTRLPROCESS_HPP
#define TDV_CTRLPROCESS_HPP

#include <cv.h>
#include <tdvbasic/common.hpp>
#include "process.hpp"
#include "pipe.hpp"

TDV_NAMESPACE_BEGIN

class CtrlProcess: public Process
{
public:
    CtrlProcess();
    
    void inputs(ReadPipe<CvMat*> *lrpipe, ReadPipe<CvMat*> *rrpipe)
    {
        m_lrpipe = lrpipe;
        m_rrpipe = rrpipe;
    }
    
    ReadPipe<CvMat*>* leftImgOutput() 
    {
        return &m_lwpipe;
    }
    
    ReadPipe<CvMat*>* rightImgOutput() 
    {
        return &m_rwpipe;
    }

    void process();
    
    void pause()
    {
        m_step = false;
    }
    
    void step()
    {
        m_mode = Step;
        m_step = true;
    }
    
    void continuous()
    {
        m_step = true;
        m_mode = Continuous;
    }
    
private:
    enum Mode
    {
        Continuous, Step
    };
    
    ReadPipe<CvMat*> *m_lrpipe, *m_rrpipe;
    ReadWritePipe<CvMat*> m_lwpipe, m_rwpipe;
    bool m_step;
    Mode m_mode;
};

TDV_NAMESPACE_END

#endif /* TDV_CTRLPROCESS_HPP */
