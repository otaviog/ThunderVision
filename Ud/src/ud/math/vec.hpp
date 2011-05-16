#ifndef UD_VEC_HPP
#define UD_VEC_HPP

#include <cmath>
#include <ostream>
#include "../common.hpp"

#include "vec2.hpp"
#include "vec3.hpp"
#include "vec4.hpp"

UD_NAMESPACE_BEGIN

inline Vec3f cosineInterpolate(const Vec3f &orign, const Vec3f &dest, float u)
{
    u = (1.0f - std::cos(u*UD_FPI))/2.0f;
    return orign * u + dest * (1 - u);
}

inline bool operator==(const Vec3f &lfs, const Vec3f &rhs)
{
    return lfs.v[0] == rhs.v[0] && lfs[1] == rhs.v[1] && lfs.v[2] == rhs.v[2];
}

template<typename VecType>
inline VecType vecLinearInterpolate(const VecType &orign, const VecType &dest, float u)
{
    return orign*(1.0f - u) + dest*u;
}

inline Vec3f lineLinearInterpolate(const Vec3f &p1, const Vec3f &p2, float u)
{
    return vecLinearInterpolate(p1, p2-p1, u);
}

inline Vec3f vecProj(const Vec3f &u, const Vec3f &v)
{
    return (vecDot(u, v) / vecDot(u, u)) * u;
}

inline void vecOrthogonalize3(const Vec3f &v1, Vec3f *v2, Vec3f *v3)
{
    *v2 = *v2 - vecProj(v1, *v2);
    *v3 = *v3 - vecProj(v1, *v3) - vecProj(*v2, *v3);
}

inline std::ostream& operator<<(std::ostream &out, const Vec3f &v)
{
    out<<v.v[0]<<" "<<v.v[1]<<" "<<v.v[2];
    return out;
}

inline std::ostream& operator<<(std::ostream &out, const Vec4f &v)
{
    out<<v[0]<<" "<<v[1]<<" "<<v[2]<<" "<<v[3];
    return out;
}

template<typename CastOut, typename CastIn>
CastOut vec_cast(const CastIn &v);

template<>
inline Vec3f vec_cast(const Vec4f &v)
{
    return Vec3f(v[0], v[1], v[2]);
}

template<>
inline Vec4f vec_cast(const Vec3f &v)
{
    return Vec4f(v[0], v[1], v[2], 1.0f);
}

UD_NAMESPACE_END

#endif
