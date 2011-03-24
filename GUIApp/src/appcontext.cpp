#include <tdvision/thunderlang.hpp>
#include "appcontext.hpp"

void AppContext::init(ExceptionReport *report)
{
    ThunderSpec spec;
    ThunderLangParser parser(spec);
    
    parser.parseFilename(m_specfname);
    
    if ( m_procRunner != NULL )
    {
        m_inputProc = m_inputFactory->create();
        m_inputLeftTee.input(m_inputProc->leftImgOutput());
        m_inputRightTee.input(m_inputProc->rightImgOutput());
    
        tdv::Process *procs[] = { 
            &m_inputLeftTee, 
            &m_inputRightTee,
            NULL }
    
        m_procRunner = new tdv::ProcessRunner(procs, report);
        m_procRunner->run();
    
        m_inputProc->init();
    }    
}

void AppContext::dispose()
{
    if ( m_procRunner != NULL )
    {
        m_inputProc->dispose();
        delete m_inputProc;
        m_inputProc = NULL;
        
        m_procRunner->join();
    }
}

void AppContext::switchInputs()
{
    m_inputLeftTee.input(m_inputProc->rightImgOutput());
    m_inputRightTee.input(m_inputProc->leftImgOutput());
}

ReadPipeTuple<IplImage*> AppContext::enableSourceImages()
{    
    m_inputLeftTee.enableOutput2();    
    m_inputRightTee.enableOutput2();
    
    return ReadPipeTuple<IplImage>(m_inputLeftTee.output2(), m_inputRightTee.output2());
}
    
void AppContext::disableSourceImages()
{
    m_inputLeftTee.disableOutput2();
    m_inputRightTee.disableOutput2();
}

CalibrationContext* AppContext::createCalibrationContext()
{
    if ( m_calibCtx != NULL )
    {
        m_calibCtx = new CalibrationContext(15);        
        m_calibCtx->init(m_inputLeftTee.output(),
                         m_inputRightTee.output());
        
        return m_calibCtx;
    }
}

void AppContext::disposeCalibrationContext()
{
    m_calib->dispose();
}
