#ifndef TDV_CALIBRATION_HPP
#define TDV_CALIBRATION_HPP

#include <tdvbasic/common.hpp>
#include <cv.h>
#include "chessboardpattern.hpp"
#include "camerasdesc.hpp"
#include "workunit.hpp"
#include "pipe.hpp"
#include "tmpbufferimage.hpp"
#include "process.hpp"

TDV_NAMESPACE_BEGIN

class Calibration;

class CalibrationObserver
{
public:
    virtual void calibrationUpdate(const Calibration &calib) = 0;
    
private:
};

class Calibration: public WorkUnit
{
public:
    Calibration(size_t numFrames);
    
    void chessPattern(const ChessboardPattern &cbpattern);
    
    void input(ReadPipe<CvMat*> *rlpipe, ReadPipe<CvMat*> *rrpipe)               
    {
        m_rlpipe = rlpipe;
        m_rrpipe = rrpipe;
    }
    
    ReadPipe<CvMat*>* detectionImage()
    {
        return &m_dipipe;
    }
    
    size_t framesProcessed() const
    {
        return m_avalFrames;
    }
    
    size_t numFrames() const
    {
        return m_numFrames;
    }

    void observer(CalibrationObserver *obsr)
    {
        m_observer = obsr;
    }
    
    const CamerasDesc& camerasDesc() const
    {
        return m_camDesc;
    }
    
    bool update();
    
private:

    CvMat* updateChessboardCorners(const CvMat *limg, 
                                   const CvMat *rimg);
    
    void updateCalibration(const CvSize &imgSize);
    
    ChessboardPattern m_cbpattern;
    
    ReadPipe<CvMat*> *m_rlpipe, *m_rrpipe;
    ReadWritePipe<CvMat*> m_dipipe;
    
    TmpBufferImage m_limg, m_rimg;
    
    CamerasDesc m_camDesc;
    
    std::vector<CvPoint2D32f> m_lPoints, m_rPoints;
    std::vector<CvPoint3D32f> m_objPoints;
    
    size_t m_numFrames, m_avalFrames, m_currFrame;

    CalibrationObserver *m_observer;        
};

class CalibrationProc: public Process, public Calibration
{
public:
    CalibrationProc(size_t maxFrames)
        : Calibration(maxFrames)
    {
    }
    
    void process()
    {
        while ( update() )
        { }
    }
};

TDV_NAMESPACE_END

#endif /* TDV_CALIBRATION_HPP */
