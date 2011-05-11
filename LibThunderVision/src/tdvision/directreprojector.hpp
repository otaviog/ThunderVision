#ifndef TDV_DIRECTREPROJECTOR_HPP
#define TDV_DIRECTREPROJECTOR_HPP

#include <tdvbasic/common.hpp>
#include "reprojector.hpp"

TDV_NAMESPACE_BEGIN

class DirectReprojector: public Reprojector
{
public:
    Vec3f reproject(int x, int y, float disp) const
    {
        return Vec3f(x, y, disp);
    }    
};

TDV_NAMESPACE_END

#endif /* TDV_DIRECTREPROJECTOR_HPP */
