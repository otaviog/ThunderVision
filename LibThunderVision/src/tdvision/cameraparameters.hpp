#ifndef TDV_CAMERAPARAMETERS_HPP
#define TDV_CAMERAPARAMETERS_HPP

#include <tdvbasic/common.hpp>
#include <cv.h>

TDV_NAMESPACE_BEGIN

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

TDV_NAMESPACE_END

#endif /* TDV_CAMERAPARAMETERS_HPP */
