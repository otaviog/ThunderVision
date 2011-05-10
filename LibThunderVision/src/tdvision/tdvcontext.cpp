#include "stereoinputsource.hpp"
#include "thunderlang.hpp"
#include "reconstruction.hpp"
#include "calibration.hpp"
#include "devstereomatcher.hpp"
#include "cpustereomatcher.hpp"
#include "ssddev.hpp"
#include "wtadev.hpp"
#include "processgroup.hpp"
#include "commonstereomatcherfactory.hpp"
#include "tdvcontext.hpp"

TDV_NAMESPACE_BEGIN

static ThunderSpec g_defaultSpec;

TDVContext::TDVContext()
{
    m_inputSrc = NULL;
    m_inputRunner = NULL;
    m_reconstRunner = NULL;
    m_calibRunner = NULL;
    m_errHandler = NULL;
    m_matcher = NULL;
    spec(NULL);
}

void TDVContext::spec(tdv::ThunderSpec *spec)
{
    if ( spec == NULL )
    {
        m_spec = &g_defaultSpec;
        return ;
    }
    
    m_spec = spec;
}

void TDVContext::start(StereoInputSource *inputSrc)
{
    if ( m_inputRunner == NULL )
    {
        assert(inputSrc != NULL);
        m_inputSrc = inputSrc;

        m_inputTees[0].input(m_inputSrc->leftImgOutput());
        m_inputTees[1].input(m_inputSrc->rightImgOutput());

        ArrayProcessGroup pgrp;
        pgrp.addProcess(*m_inputSrc);
        pgrp.addProcess(&m_inputTees[0]);
        pgrp.addProcess(&m_inputTees[1]);

        m_inputRunner = new ProcessRunner(pgrp, this);
        m_inputRunner->run();
    }
}

void TDVContext::dispose()
{
    if ( m_inputRunner != NULL )
    {
        m_inputRunner->finishAll();
        m_inputRunner->join();
        delete m_inputRunner;
        m_inputRunner = NULL;
    }
}

Reconstruction* TDVContext::runReconstruction(const std::string &profileName)
{
    Reconstruction *reconst = NULL;

    if ( m_reconstRunner != NULL )
    {
        return reconst;
    }
    
    CommonStereoMatcherFactory matcherFactory;
    
    if ( profileName == "Device" )
    {
        matcherFactory.computeDev(CommonStereoMatcherFactory::Device);
        matcherFactory.maxDisparity(128);
        matcherFactory.optimization(CommonStereoMatcherFactory::WTA);
        //matcherFactory.matchingCost(CommonStereoMatcherFactory::CrossCorrelationNorm);
    }
    else if ( profileName == "CPU" )
    {
        matcherFactory.computeDev(CommonStereoMatcherFactory::CPU);
    }
            
    m_matcher = matcherFactory.createStereoMatcher();

    m_inputTees[0].enable(RECONSTRUCTION_TEE_ID);
    m_inputTees[1].enable(RECONSTRUCTION_TEE_ID);    
    
    reconst = new Reconstruction(m_matcher,
                                 m_inputTees[0].output(RECONSTRUCTION_TEE_ID),
                                 m_inputTees[1].output(RECONSTRUCTION_TEE_ID));
    
    reconst->camerasDesc(m_spec->camerasDesc("default"));
    reconst->benchmarkCallback(this);
    m_reconstRunner = new ProcessRunner(*reconst, this);
    m_reconstRunner->run();
    
    return reconst;
}

void TDVContext::releaseReconstruction(Reconstruction *reconst)
{
    if ( m_reconstRunner != NULL )
    {
        m_inputTees[0].disable(RECONSTRUCTION_TEE_ID);
        m_inputTees[1].disable(RECONSTRUCTION_TEE_ID);
    
        m_reconstRunner->join();

        delete m_reconstRunner;
        m_reconstRunner = NULL;
        delete reconst;        
    }
}

Calibration* TDVContext::runCalibration()
{
    CalibrationProc *calib = NULL;
    if ( m_calibRunner == NULL )
    {                
        calib = new CalibrationProc(3);
        
        m_inputTees[0].enable(CALIBRATION_TEE_ID);
        m_inputTees[1].enable(CALIBRATION_TEE_ID);
        
        calib->input(m_inputTees[0].output(CALIBRATION_TEE_ID),
                     m_inputTees[1].output(CALIBRATION_TEE_ID));
        
        ArrayProcessGroup grp;
        grp.addProcess(calib);
        m_calibRunner = new ProcessRunner(grp, this);
        m_calibRunner->run();                
    }
    
    return calib;
}

void TDVContext::releaseCalibration(Calibration *calib)
{
    assert(calib != NULL);
    if ( m_calibRunner != NULL )
    {        
        m_inputTees[0].disable(CALIBRATION_TEE_ID);
        m_inputTees[1].disable(CALIBRATION_TEE_ID);
        
        m_calibRunner->join();
        if ( calib->isComplete() )
        {
            m_spec->camerasDesc("default") = calib->camerasDesc();
        }
        
        delete calib;
        
        m_calibRunner = NULL;
    }
}

void TDVContext::errorOcurred(const std::exception &err)
{
    if ( m_errHandler != NULL )
    {
        m_errHandler->errorOcurred(err);
    }
}

void TDVContext::dupInputSource(
    ReadPipe<CvMat*> **leftSrc,
    ReadPipe<CvMat*> **rightSrc)
{
    m_inputTees[0].enable(VIEW_TEE_ID);
    m_inputTees[1].enable(VIEW_TEE_ID);

    *leftSrc = m_inputTees[0].output(VIEW_TEE_ID);
    *rightSrc = m_inputTees[1].output(VIEW_TEE_ID);
}

void TDVContext::undupInputSource()
{
    m_inputTees[0].disable(VIEW_TEE_ID);
    m_inputTees[1].disable(VIEW_TEE_ID);
}

void TDVContext::switchCameras()
{
    m_inputTees[0].waitPauseProc();
    m_inputTees[1].waitPauseProc();
    
    ReadPipe<CvMat*> *aux = m_inputTees[0].input();
    m_inputTees[0].input(m_inputTees[1].input());
    m_inputTees[1].input(aux);    

    m_inputTees[0].resumeProc();
    m_inputTees[1].resumeProc();    
}

void TDVContext::reconstructionDone(float framesSec)
{
    m_inputSrc->framesPerSec(framesSec);
}

TDV_NAMESPACE_END
