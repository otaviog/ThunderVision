#include "ctrlprocess.hpp"

TDV_NAMESPACE_BEGIN

CtrlProcess::CtrlProcess()
{
    m_step = false;
    m_mode = Step;
}

void CtrlProcess::process()
{
    CvMat *limg, *rimg;
    
    while ( m_lrpipe->read(&limg) && m_rrpipe->read(&rimg) )
    {
        if ( m_step )
        {
            m_lwpipe.write(limg);
            m_rwpipe.write(rimg);

            if ( m_mode == Step )
                m_step = false;
        }
        else
        {
            cvDecRefData(limg);
            cvDecRefData(rimg);
        }
    }    

    m_lwpipe.finish();
    m_rwpipe.finish();
}

TDV_NAMESPACE_END
