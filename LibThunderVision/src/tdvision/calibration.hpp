#ifndef TDV_CALIBRATION_HPP
#define TDV_CALIBRATION_HPP

#include <tdvbasic/common.hpp>
#include <cv.h>
#include "chessboardpattern.hpp"
#include "cameraparameters.hpp"
#include "workunit.hpp"
#include "pipe.hpp"

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
    
    void input(ReadPipe<IplImage*> *rlpipe, ReadPipe<IplImage*> *rrpipe,
               bool sinkLeft, bool sinkRight)
    {
        m_rlpipe = rlpipe;
        m_rrpipe = rrpipe;
        m_sinkLeft = sinkLeft;
        m_sinkRight = sinkRight;
    }
    
    ReadPipe<IplImage*>* detectionImage()
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
    
    const CameraParameters& leftCamParms() const
    {
        return m_lParms;
    }
    
    const CameraParameters& rightCamParms() const
    {
        return m_rParms;
    }
        
    bool update();
    
private:

    IplImage* updateChessboardCorners(const IplImage *limg, const IplImage *rimg);
    
    void updateCalibration(const CvSize &imgSize);
    
    ChessboardPattern m_cbpattern;
    
    ReadPipe<IplImage*> *m_rlpipe, *m_rrpipe;
    bool m_sinkLeft, m_sinkRight;    
    ReadWritePipe<IplImage*> m_dipipe;
    
    CameraParameters m_lParms, m_rParms;    
    std::vector<CvPoint2D32f> m_lPoints, m_rPoints;
    std::vector<CvPoint3D32f> m_objPoints;
    
    size_t m_numFrames, m_avalFrames, m_currFrame;

    CalibrationObserver *m_observer;
};
TDV_NAMESPACE_END

#endif /* TDV_CALIBRATION_HPP */
