#ifndef TDV_CALIBRATION_HPP
#define TDV_CALIBRATION_HPP

#include <tdvbasic/common.hpp>
#include <cv.h>
#include "workunit.hpp"
#include "pipe.hpp"

TDV_NAMESPACE_BEGIN

class ChessboardPattern
{
public:
    ChessboardPattern();
        
    CvSize dim()
    {
        return m_dim;
    }
    
    size_t totalCorners() const
    {
        return m_dim.width*m_dim.height;
    }
    
    void generateObjectPoints(std::vector<CvPoint3D32f> &v) const;
    
private:       
    CvSize m_dim;    
    float m_squareSize;
};

class CameraParameters
{
public:
    CameraParameters();
    
    ~CameraParameters()
    {
        
    }
    
    CvMat intrinsecs() 
    {
        return cvMat(3, 3, CV_64F, m_intrinsecs);
    }
    
    CvMat distorsion()
    {
        return cvMat(1, 5, CV_64F, m_distorsion);
    }
    
private:
    double m_intrinsecs[3][3];
    double m_distorsion[5];
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
    
    bool update();
        
private:

    IplImage* updateChessboardCorners(IplImage *limg, IplImage *rimg);
    
    void updateCalibration();
    
    ChessboardPattern m_cbpattern;
    
    ReadPipe<IplImage*> *m_rlpipe, *m_rrpipe;
    bool m_sinkLeft, m_sinkRight;    
    ReadWritePipe<IplImage*> m_dipipe;
    
    CameraParameters m_lParms, m_rParms;    
    std::vector<CvPoint2D32f> m_lPoints, m_rPoints;
    std::vector<CvPoint3D32f> m_objPoints;
    
    size_t m_numFrames, m_frameCount;
};
TDV_NAMESPACE_END

#endif /* TDV_CALIBRATION_HPP */
