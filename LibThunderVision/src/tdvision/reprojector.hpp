#ifndef TDV_REPROJECTOR_HPP
#define TDV_REPROJECTOR_HPP

#include <tdvbasic/common.hpp>
#include <ud/math/vec3.hpp>
#include "dim.hpp"

TDV_NAMESPACE_BEGIN

class Reprojector
{
public:
    virtual ud::Vec3f reproject(int x, int y, float disp, const Dim &imgDim) const = 0;
};

TDV_NAMESPACE_END

#endif /* TDV_REPROJECTOR_HPP */
