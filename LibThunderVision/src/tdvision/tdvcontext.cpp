#include "stereoinputsource.hpp"
#include "thunderlang.hpp"
#include "reconstruction.hpp"
#include "devstereomatcher.hpp"
#include "cpustereomatcher.hpp"
#include "ssddev.hpp"
#include "wtadev.hpp"
#include "tdvcontext.hpp"

TDV_NAMESPACE_BEGIN

TDVContext::TDVContext()
{
    m_inputSrc = NULL;
    m_runner = NULL;
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
    if ( m_runner == NULL )
    {
        assert(inputSrc != NULL);    
        m_inputSrc = inputSrc;

        m_inputLeftTee.input(m_inputSrc->leftImgOutput());
        m_inputRightTee.input(m_inputSrc->rightImgOutput());
        
        ArrayProcessGroup pgrp;
        pgrp.addProcess(*m_inputSrc);
        pgrp.addProcess(&m_inputLeftTee);
        pgrp.addProcess(&m_inputRightTee);
        
        m_runner = new ProcessRunner(pgrp, this);
        m_runner->run();
    }
}

void TDVContext::dispose()
{
    if ( m_runner != NULL )
    {
        m_runner->finishAll();
        m_runner->join();
        delete m_runner;
        m_runner = NULL;
    }
}

Reconstruction* TDVContext::runReconstruction(const std::string &profileName)
{
    Reconstruction *reconst = NULL;
    
    if ( m_reconstRunner == NULL )
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
        
        reconst = new Reconstruction(m_matcher, m_inputLeftTee.output1(),
                                     m_inputLeftTee.output2());
        
        
    }
    else if ( profileName == "CPU" )
    {
        m_matcher = new CPUStereoMatcher();
        reconst = new Reconstruction(m_matcher, m_inputLeftTee.output1(),
                                     m_inputLeftTee.output2());

    }
    
    m_reconstRunner = new ProcessRunner(*reconst, this);
    m_reconstRunner->run();
    
    return reconst;
}
    
void TDVContext::releaseReconstruction(Reconstruction *reconst)
{
    if ( m_reconstRunner != NULL ) 
    {
        m_reconstRunner->finishAll();
        m_reconstRunner->join();
        delete m_reconstRunner;
    }
}

Calibration* TDVContext::runCalibration()
{
    return NULL;
}

void TDVContext::errorOcurred(const std::exception &err)
{
    if ( m_errHandler != NULL )
    {
        m_errHandler->errorOcurred(err);
    }
}

void TDVContext::dupInputSource(
    ReadPipe<IplImage*> **leftSrc, 
    ReadPipe<IplImage*> **rightSrc)
{
    m_inputLeftTee.enableOutput2();    
    m_inputRightTee.enableOutput2();
        
    *leftSrc = m_inputLeftTee.output2();
    *rightSrc = m_inputRightTee.output2();
}
    
void TDVContext::undupInputSource()
{
    m_inputLeftTee.disableOutput2();
    m_inputRightTee.disableOutput2();
}

TDV_NAMESPACE_END
