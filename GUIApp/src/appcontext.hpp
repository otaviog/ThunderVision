#ifndef TDV_APPCONTEXT_HPP
#define TDV_APPCONTEXT_HPP

#include <string>
#include <tdvision/pipe.hpp>
#include "inputprocess.hpp"

template<typename Type>
struct ReadPipeTuple
{
    tdv::ReadPipe<Type> *p1;
    tdv::ReadPipe<Type> *p2;
};

class AppContext
{
public:
    void specFilename(const std::string &filename)
    {
        m_specfname = filename;
    }
    
    void inputProcessFactory(const InputProcessFactory *factory);
    
    void init();
    
    void dispose();
    
    ReadPipeTuple<IplImage*> enableSourceImages();
        
    void disableSourceImages();
        
    void switchCameras();        
    
    CalibrationContext* createCalibrationContext();
    
    void disposeCalibrationContext();
    
    //ReconstructionContext* reconstructionContext();
    
private:    
    tdv::ProcessRunner *m_runner;    
    InputProcessFactory *m_inputFactory;
    InputProcess *m_inputProc;
    
    tdv::WorkUnitProcess<tdv::TeeWorkUnit<IplImage*> > m_inputLeftTee;
    tdv::WorkUnitProcess<tdv::TeeWorkUnit<IplImage*> > m_inputRightTee;
    
    std::string m_specfname;
    
    
    CalibrationContext *m_calibCtx;
};

#endif /* TDV_APPCONTEXT_HPP */
