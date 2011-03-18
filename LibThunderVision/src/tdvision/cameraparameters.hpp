#ifndef TDV_CAMERAPARAMETERS_HPP
#define TDV_CAMERAPARAMETERS_HPP

#include <tdvbasic/common.hpp>
#include <cv.h>
#include <ostream>

TDV_NAMESPACE_BEGIN

class CameraParameters
{
public:
    CameraParameters();
        
    CameraParameters(const CameraParameters &cp)
    {
        copy(cp);
    }
    
    CameraParameters& operator=(const CameraParameters &cp)
    {
        copy(cp);
        return *this;
    }
    
    const CvMat& intrinsecs() const
    {
        return m_vi;
    }
    
    const CvMat& distorsion() const
    {
        return m_vd;
    }

    CvMat& intrinsecs()
    {
        return m_vi;
    }
    
    CvMat& distorsion()
    {
        return m_vd;
    }

private:
    
    void copy(const CameraParameters &cp);
    
    double m_intrinsecs[3][3];
    double m_distorsion[5];
    CvMat m_vi, m_vd;    
};

std::ostream& operator<<(std::ostream& out, const CameraParameters &cp);

TDV_NAMESPACE_END

#endif /* TDV_CAMERAPARAMETERS_HPP */
