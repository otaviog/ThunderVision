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
    
    bool hasFundamentalMatrix() const
    {
        return m_hasF;
    }

    const double* fundamentalMatrix() const
    {
        return m_F;
    }

    void extrinsics(const double R[9], const double T[3]);
    
    const double* extrinsicsR() const
    {
        return m_extrinsicsR;
    }
    
    const double* extrinsicsT() const
    {
        return m_extrinsicsT;
    }
    
    bool hasExtrinsics() const
    {
        return m_hasExt;
    }

private:
    CameraParameters m_leftCam, m_rightCam;
    double m_F[9], m_extrinsicsR[9], m_extrinsicsT[3];
    bool m_hasF, m_hasExt;
    
};

TDV_NAMESPACE_END

#endif /* TDV_CAMERASDESC_HPP */
