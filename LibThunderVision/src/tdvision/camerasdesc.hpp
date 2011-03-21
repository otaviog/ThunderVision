#ifndef TDV_CAMERASDESC_HPP
#define TDV_CAMERASDESC_HPP

#include <string>
#include <tdvbasic/common.hpp>
#include "cameraparameters.hpp"

TDV_NAMESPACE_BEGIN

class CamerasDesc
{
public:
    CamerasDesc();
    
    CameraParameters& leftCamera()
    {
        return m_leftCam;
    }
    
    CameraParameters& rightCamera()
    {
        return m_rightCam;
    }
    
    const CameraParameters& leftCamera() const
    {
        return m_leftCam;
    }
    
    const CameraParameters& rightCamera() const
    {
        return m_rightCam;
    }

    void fundamentalMatrix(const double mtx[9]);
    
    const double* fundamentalMatrix() const
    {
        return m_F;
    }
    
private:
    CameraParameters m_leftCam, m_rightCam;
    double m_F[9];    
};

TDV_NAMESPACE_END

#endif /* TDV_CAMERASDESC_HPP */
