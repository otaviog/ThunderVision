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
    
    const CvPoint3D32f* objectPoints() const
    {
        return m_objPoints;
    }
    
private:
    CvSize m_dim;
    CvPoint3D32f *m_objPoints;
    float m_squareSize;
};

class CameraParameters
{
public:
    
private:
    
};

class Calibration: public WorkUnit
{
public:
    Calibration();
    
    void input(ReadPipe<IplImage*> *rlpipe, ReadPipe<IplImage*> *rrpipe)
    {
        m_rlpipe = rlpipe;
        m_rrpipe = rrpipe;
    }
    
    ReadPipe<IplImage*>* detectionImage()
    {
        return &m_dipipe;
    }
    
    bool update();
        
private:
    ReadPipe<IplImage*> *m_rlpipe, *m_rrpipe;
    ReadWritePipe<IplImage*> m_dipipe;
    
    double m_intrinsecs[3][3];
    double m_distorsion[5];
    
    ChessboardPattern m_chessboard;
    
};
TDV_NAMESPACE_END

#endif /* TDV_CALIBRATION_HPP */
