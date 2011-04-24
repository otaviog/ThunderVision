#include "sink.hpp"
#include "ctrlprocess.hpp"

TDV_NAMESPACE_BEGIN

CtrlWork::CtrlWork()
{
    m_step = false;
    m_mode = Step;    
    m_hasWrite = false;
}

bool CtrlWork::update()
{
    CvMat *limg, *rimg;
    
    try
    {
        if ( m_lrpipe->read(&limg) && m_rrpipe->read(&rimg) )
        {
            if ( m_step )
            {
                m_lwpipe.write(limg);
                m_rwpipe.write(rimg);

                m_hasWrite = true;
                if ( m_mode == Step )
                    m_step = false;
            }
            else
            {
                CvMatSinkPol::sink(limg);
                CvMatSinkPol::sink(rimg);
                m_hasWrite = false;
            }
            
            return true;
        }
        else
        {
            m_lwpipe.finish();
            m_rwpipe.finish();
            
            return false;
        }
    }
    catch (const std::exception &ex)
    {
        m_lwpipe.finish();
        m_rwpipe.finish();
        throw ex;                
    }
}

TDV_NAMESPACE_END
