#include "cameraparameters.hpp"

TDV_NAMESPACE_BEGIN

CameraParameters::CameraParameters()
{
    m_vi = cvMat(3, 3, CV_64F, m_intrinsecs);
    m_vd = cvMat(1, 5, CV_64F, m_distorsion);
    cvSetIdentity(&m_vi);
    cvSetIdentity(&m_vd);
}

void CameraParameters::copy(const CameraParameters &cp)
{
    memcpy(m_intrinsecs, cp.m_intrinsecs, 
           sizeof(double)*9);
    memcpy(m_distorsion, cp.m_distorsion,
           sizeof(double)*5);
    
    m_vi = cvMat(3, 3, CV_64F, m_intrinsecs);
    m_vd = cvMat(1, 5, CV_64F, m_distorsion);        
}

std::ostream& operator<<(std::ostream& out, const CameraParameters &cp)
{
    const CvMat &M = cp.intrinsecs();
    const CvMat &D = cp.distorsion();
        
    out << "[[" << M.data.db[0] << ", " << M.data.db[4] << ", " << M.data.db[8]
        << ", " << M.data.db[2] << ", " << M.data.db[5]
        << "], [" 
        << D.data.db[0] << ", " << D.data.db[1] << ", " << D.data.db[2]
        << ", " << D.data.db[3] << ", " << D.data.db[4] << "]]";
    
    return out;
}

TDV_NAMESPACE_END
