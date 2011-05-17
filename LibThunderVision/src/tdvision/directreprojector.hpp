#ifndef TDV_DIRECTREPROJECTOR_HPP
#define TDV_DIRECTREPROJECTOR_HPP

#include <tdvbasic/common.hpp>
#include "reprojector.hpp"

TDV_NAMESPACE_BEGIN

class DirectReprojector: public Reprojector
{
public:    
    ud::Vec3f reproject(int x, int y, float disp, const Dim &imgDim) const
    {
        return ud::Vec3f(x, imgDim.height() - y, disp);
    }    
};

TDV_NAMESPACE_END

#endif /* TDV_DIRECTREPROJECTOR_HPP */
