#ifndef TDV_TDVCONTEXT_HPP
#define TDV_TDVCONTEXT_HPP

#include <tdvbasic/common.hpp>
#include <string>
#include <cv.h>
#include "teeworkunit.hpp"
#include "workunitprocess.hpp"
#include "processrunner.hpp"
#include "exceptionreport.hpp"

TDV_NAMESPACE_BEGIN

class StereoInputSource;
class StereoMatcher;
class Reconstruction;
class Calibration;
class ThunderSpec;

class TDVContext: public ExceptionReport
{
public:    
    TDVContext();
    
    void spec(tdv::ThunderSpec *spec);

    void start(StereoInputSource *inputSrc);    
    
    void dispose();
    
    Reconstruction* runReconstruction(const std::string &profileName);    
    
    void releaseReconstruction(Reconstruction *reconst);
    
    Calibration* runCalibration();

    void dupInputSource(ReadPipe<CvMat*> **leftSrc, 
                        ReadPipe<CvMat*> **rightSrc);
    
    void undupInputSource();
    
    void errorOcurred(const std::exception &err);
    
    void errorHandler(ExceptionReport *handler)
    {
        m_errHandler = handler;
    }    
    
private:
    void specChanged();
    
    tdv::ProcessRunner *m_runner, *m_reconstRunner;
    
    StereoInputSource *m_inputSrc;
    TWorkUnitProcess<TeeWorkUnit<CvMat*, CvMatSinkPol> > m_inputTees[2];
    
    StereoMatcher *m_matcher;
    
    ExceptionReport *m_errHandler;
        
    ThunderSpec *m_spec;
};

TDV_NAMESPACE_END

#endif /* TDV_TDVCONTEXT_HPP */
