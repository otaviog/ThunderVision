#ifndef TDV_CAMERAPARAMETERS_HPP
#define TDV_CAMERAPARAMETERS_HPP

#include <tdvbasic/common.hpp>
#include <ostream>

TDV_NAMESPACE_BEGIN

class CameraParameters
{
public:
    CameraParameters();
                
    void distortion(double d1, double d2, double d3, 
                    double d4, double d5)
    {
        m_distortion[0] = d1;
        m_distortion[1] = d2;
        m_distortion[2] = d3;
        m_distortion[3] = d4;
        m_distortion[4] = d5;
    }
    
    const double *distortion() const
    {
        return m_distortion;
    }

    void intrinsics(double mtx[9]);
            
    const double* intrinsics() const
    {
        return m_intrinsics;
    }
        
private:        
    double m_intrinsics[9];    
    double m_distortion[5];        
};

std::ostream& operator<<(std::ostream& out, const CameraParameters &cp);

TDV_NAMESPACE_END

#endif /* TDV_CAMERAPARAMETERS_HPP */
