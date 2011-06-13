#include "sink.hpp"
#include "ctrlprocess.hpp"

TDV_NAMESPACE_BEGIN

FlowCtrl::FlowCtrl()
{
    m_step = false;
    m_mode = Step;    
    m_hasWrite = false;
}

bool FlowCtrl::testFlow()
{
    if ( m_step )
    {                
        if ( m_mode == Step )
            m_step = false;
        
        return true;
    }
    
    return false;
}


bool CtrlWork::update()
{
    CvMat *limg = NULL, *rimg = NULL;
    
    try
    {
        if ( m_lrpipe->read(&limg) && m_rrpipe->read(&rimg) )
        {            
            if ( testFlow() )
            {
                if ( (mode() == Continuous 
                      && !m_lwpipe.isFull() 
                      && !m_rwpipe.isFull()) 
                     || mode() != Continuous )
                {
                        m_lwpipe.write(limg);
                        m_rwpipe.write(rimg);                    
                }
                else
                {
                    CvMatSinkPol::sink(limg);
                    CvMatSinkPol::sink(rimg);
                }
            }
            else
            {
                CvMatSinkPol::sink(limg);
                CvMatSinkPol::sink(rimg);
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
