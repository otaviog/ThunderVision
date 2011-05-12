#ifndef TDV_CVREPROJECTOR_HPP
#define TDV_CVREPROJECTOR_HPP

#include <tdvbasic/common.hpp>
#include <cv.h>
#include "reprojector.hpp"

TDV_NAMESPACE_BEGIN

class CVReprojector: public Reprojector
{
public:
    
    CVReprojector();
    
    Vec3f reproject(int x, int y, float disp) const;    
    
    void qmatrix(double mtx[16])
    {
        memcpy(m_qMatrix, mtx, sizeof(double)*16);
    }
    
private:
    mutable double m_qMatrix[16];
};

TDV_NAMESPACE_END

#endif /* TDV_CVREPROJECTOR_HPP */
