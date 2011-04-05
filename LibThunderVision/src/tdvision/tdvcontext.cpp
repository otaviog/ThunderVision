#include "stereoinputsource.hpp"
#include "thunderlang.hpp"
#include "reconstruction.hpp"
#include "calibration.hpp"
#include "devstereomatcher.hpp"
#include "cpustereomatcher.hpp"
#include "ssddev.hpp"
#include "wtadev.hpp"
#include "processgroup.hpp"
#include "tdvcontext.hpp"

TDV_NAMESPACE_BEGIN

TDVContext::TDVContext()
{
    m_inputSrc = NULL;
    m_inputRunner = NULL;
    m_errHandler = NULL;
    m_matcher = NULL;
}

void TDVContext::loadSpecFromFile(const std::string &specfilename)
{

    ThunderSpec spec;
    ThunderLangParser(spec).parseFile(specfilename);
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

    if ( profileName == "Device" )
    {
        DevStereoMatcher *matcher = new DevStereoMatcher;
        m_matcher = matcher;
        matcher->setMatchingCost(boost::shared_ptr<SSDDev>(
                                     new SSDDev(255, 1024*1024*256)));
        matcher->setOptimizer(boost::shared_ptr<WTADev>(new WTADev));
    }
    else if ( profileName == "CPU" )
    {
        m_matcher = new CPUStereoMatcher();
    }

    m_inputTees[0].enable(0);
    m_inputTees[1].enable(0);

    reconst = new Reconstruction(m_matcher,
                                 m_inputTees[0].output(0),
                                 m_inputTees[1].output(0));
    
    m_reconstRunner = new ProcessRunner(*reconst, this);
    m_reconstRunner->run();

    return reconst;
}

void TDVContext::releaseReconstruction(Reconstruction *reconst)
{
    if ( m_reconstRunner != NULL )
    {
        m_inputTees[0].disable(0);
        m_inputTees[1].disable(0);
    
        m_reconstRunner->join();

        delete m_reconstRunner;
        m_reconstRunner = NULL;
        delete reconst;        
    }
}

Calibration* TDVContext::runCalibration()
{
    CalibrationProc *calib = NULL;
    if ( m_calibRunner != NULL )
    {                
        calib = new CalibrationProc(10);
        
        m_inputTees[0].enable(2);
        m_inputTees[1].enable(2);
        
        calib->input(m_inputTees[0].output(2),
                     m_inputTees[1].output(2));
        
        ArrayProcessGroup grp;
        grp.addProcess(calib);
        m_calibRunner = new ProcessRunner(grp, this);
        m_calibRunner->run();                
    }
    
    return calib;
}

void TDVContext::releaseCalibration(Calibration *calib)
{
    if ( m_calibRunner != NULL )
    {        
        m_inputTees[0].disable(0);
        m_inputTees[1].disable(0);
        
        m_calibRunner->join();
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
    m_inputTees[0].enable(1);
    m_inputTees[1].enable(1);

    *leftSrc = m_inputTees[0].output(1);
    *rightSrc = m_inputTees[1].output(1);
}

void TDVContext::undupInputSource()
{
    m_inputTees[0].disable(1);
    m_inputTees[1].disable(1);
}

TDV_NAMESPACE_END
