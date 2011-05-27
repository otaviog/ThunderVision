#ifndef TDV_DIRECTREPROJECTOR_HPP
#define TDV_DIRECTREPROJECTOR_HPP

#include <tdvbasic/common.hpp>
#include "reprojector.hpp"

TDV_NAMESPACE_BEGIN

class DirectReprojector: public Reprojector
{
public:    
    DirectReprojector(float zscale = 1.0f)
    {
        m_zscale = zscale;
    }
    
    ud::Vec3f reproject(int x, int y, float disp, const Dim &imgDim) const
    {
        return ud::Vec3f(x, imgDim.height() - y, disp*m_zscale);
    }   
    
private:
    float m_zscale;
};

TDV_NAMESPACE_END

#endif /* TDV_DIRECTREPROJECTOR_HPP */
