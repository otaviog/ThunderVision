#include "camcalibrationcontext.hpp"

CamCalibrationContext::CamCalibrationContext(size_t numFrames)
    : m_capture0(0), 
      m_capture1(1),
      m_calib(numFrames)
      
{
    m_calib.input(m_capture0.colorImage(), 
                  m_capture1.colorImage(), 
                  false, false);
    
    m_sink0.input(m_capture0.output());
    m_sink1.input(m_capture1.output());        
}

void CamCalibrationContext::init(tdv::ExceptionReport *errHdl)
{
    if ( m_procRunner == NULL )
    {
        m_calibProc = new tdv::WorkUnitProcess(m_calib);
        
        tdv::Process *procs[] = {
            &m_capture0, &m_capture1, 
            m_calibProc, &m_sink0, &m_sink1, NULL 
        };
        
        m_procRunner = new tdv::ProcessRunner(procs, errHdl);
        m_procRunner->run();
    }
}

void CamCalibrationContext::dispose()
{
    if ( m_procRunner != NULL )
    {
        m_capture0.finish();
        m_capture1.finish();
    
        m_procRunner->join();
        delete m_procRunner;
        delete m_calibProc;
        
        m_procRunner = NULL;
        m_calibProc = NULL;
    }    
}