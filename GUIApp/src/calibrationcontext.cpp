#include "calibrationcontext.hpp"

CalibrationContext::CalibrationContext(size_t numFrames)
    : m_calib(numFrames)      
{    
    m_sink0.input(m_capture0.output());
    m_sink1.input(m_capture1.output());        
}

void CalibrationContext::init(tdv::ExceptionReport *errHdl)
{
    if ( m_procRunner == NULL )
    {
        m_calibProc = new tdv::WorkUnitProcess(m_calib);
        
        tdv::Process *procs[] = {        
            m_calibProc, &m_sink0, &m_sink1, NULL 
        };
        
        m_procRunner = new tdv::ProcessRunner(procs, errHdl);
        m_procRunner->run();
    }
}

void CalibrationContext::dispose()
{
    if ( m_procRunner != NULL )
    {    
        m_procRunner->join();
        delete m_procRunner;
        delete m_calibProc;
        
        m_procRunner = NULL;
        m_calibProc = NULL;
    }    
}
