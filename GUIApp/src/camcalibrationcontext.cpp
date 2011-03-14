#include "camcalibrationcontext.hpp"

CamCalibrationContext::CamCalibrationContext()
{
    m_calib.inputLeft(m_capture0.colorImage());
    m_calib.inputRight(m_capture1.colorImage());
    
    m_sink0.input(m_capture0.output());
    m_sink1.input(m_capture1.output());        
}

void CamCalibration::start()
{
    if ( m_procRunner != NULL )
    {
        tdv::Process *procs[] = {
            &m_capture0, &m_capture1, 
            m_camWid, &m_calib,
            &m_sink0, &m_sink1, NULL };
        
        m_procRunner = new tdv::ProcessRunner(procs, this);     
        m_procRunner->run();
    }
}

void CamCalibration::stop()
{
    if ( m_procRunner != NULL )
    {
        m_capture0.finish();
        m_capture1.finish();
    
        m_procRunner->join();
        delete m_procRunner;
        
        m_procRunner = NULL;
    }    
}
