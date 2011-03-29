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

class TDVContext: public ExceptionReport
{
public:    
    TDVContext();
    
    void start(StereoInputSource *inputSrc);
    
    void loadSpecFromFile(const std::string &filename);
    
    void dispose();
    
    Reconstruction* runReconstruction(const std::string &profileName);    
    
    void releaseReconstruction(Reconstruction *reconst);
    
    Calibration* runCalibration();

    void dupInputSource(ReadPipe<IplImage*> **leftSrc, 
                        ReadPipe<IplImage*> **rightSrc);
    
    void undupInputSource();
    
    void errorOcurred(const std::exception &err);
    
    void errorHandler(ExceptionReport *handler)
    {
        m_errHandler = handler;
    }
    
private:
    tdv::ProcessRunner *m_runner, *m_reconstRunner;
    
    StereoInputSource *m_inputSrc;
    TWorkUnitProcess<TeeWorkUnit<IplImage*> > m_inputLeftTee;
    TWorkUnitProcess<TeeWorkUnit<IplImage*> > m_inputRightTee;    
    
    StereoMatcher *m_matcher;
    
    ExceptionReport *m_errHandler;
};

TDV_NAMESPACE_END

#endif /* TDV_TDVCONTEXT_HPP */
