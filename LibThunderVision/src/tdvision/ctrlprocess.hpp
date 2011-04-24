#ifndef TDV_CTRLPROCESS_HPP
#define TDV_CTRLPROCESS_HPP

#include <cv.h>
#include <tdvbasic/common.hpp>
#include "workunit.hpp"
#include "workunitprocess.hpp"
#include "process.hpp"
#include "pipe.hpp"

TDV_NAMESPACE_BEGIN

class CtrlWork: public WorkUnit
{
public:
    CtrlWork();
    
    bool update();

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
    
    bool hasWrite() const
    {
        return m_hasWrite;
    }
    
private:
    enum Mode
    {
        Continuous, Step
    };
    
    ReadPipe<CvMat*> *m_lrpipe, *m_rrpipe;
    ReadWritePipe<CvMat*> m_lwpipe, m_rwpipe;
    bool m_step, m_hasWrite;
    Mode m_mode;
};

typedef TWorkUnitProcess<CtrlWork> CtrlProcess;

TDV_NAMESPACE_END

#endif /* TDV_CTRLPROCESS_HPP */
