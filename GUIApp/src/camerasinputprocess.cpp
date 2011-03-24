#include "camerasinputprocess.hpp"

CamerasInputProcess::CamerasInputProcess()
    : m_capture0(0), m_capture1(1)
{
    m_procRunner = NULL;
}

void CamerasInputProcess::init(ExceptionReport *report)
{
    if ( m_procRunner == NULL )
    {        
        tdv::Process *procs[] = {
            &m_capture0, &m_capture1, NULL 
        };
        
        m_procRunner = new tdv::ProcessRunner(procs, report);
        m_procRunner->run();
    }
}

void CamerasInputProcess::dispose()
{   
    if ( m_procRunner != NULL )
    {
        m_capture0.finish();
        m_capture1.finish();
        
        m_procRunner.join();
        
        delete m_procRunner();
        m_procRunner = NULL;
    }        
}
