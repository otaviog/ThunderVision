#include <cstdlib>
#include <cstring>
#include "cameraparameters.hpp"

TDV_NAMESPACE_BEGIN

static void identity(double m[9])
{
    m[0] = 1.0; m[1] = 0.0; m[2] = 0.0;
    m[3] = 0.0; m[4] = 1.0; m[5] = 0.0;
    m[6] = 0.0; m[7] = 0.0; m[8] = 1.0;
}

CameraParameters::CameraParameters()
{
    m_distortion[0] 
        = m_distortion[1]
        = m_distortion[2]
        = m_distortion[3]
        = m_distortion[4] = 0.0;
    
    identity(m_intrinsics);
    identity(m_extrinsics);
}

void CameraParameters::intrinsics(double mtx[9])
{
    memcpy(m_intrinsics, mtx, sizeof(double)*9);
}
    
void CameraParameters::extrinsics(double mtx[9])
{
    memcpy(m_extrinsics, mtx, sizeof(double)*9);
}

std::ostream& operator<<(std::ostream& out, const CameraParameters &cp)
{        
    const double *M = cp.intrinsics();
    const double *D = cp.distortion();
    
    out << "[[" << M[0] << ", " << M[4] << ", " << M[8]
        << ", " << M[2] << ", " << M[5]
        << "], [" 
        << D[0] << ", " << D[1] << ", " << D[2]
        << ", " << D[3] << ", " << D[4] << "]]";
    
    return out;
}

TDV_NAMESPACE_END
