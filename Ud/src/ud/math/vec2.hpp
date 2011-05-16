#ifndef UD_VEC2_HPP
#define UD_VEC2_HPP

#include <cmath>
#include "../common.hpp"

UD_NAMESPACE_BEGIN

class Vec2f
{
public:
    Vec2f() { }

    Vec2f(float val)
    {
        v[0] = v[1] = val;
    }

    Vec2f(float _x, float _y)
    {
        v[0] = _x;
        v[1] = _y;
    }

    float operator[](int i) const
    {
        return v[i];
    }

    float& operator[](int i)
    {
        return v[i];
    }

    Vec2f& operator+=(const Vec2f &rhs)
    {
        v[0] += rhs[0];
        v[1] += rhs[1];
        return *this;
    }

    Vec2f& operator-=(const Vec2f &rhs)
    {
        v[0] -= rhs[0];
        v[1] -= rhs[1];
        return *this;
    }

    Vec2f& operator*=(float val)
    {
        v[0] *= val;
        v[1] *= val;
        return *this;
    }

    Vec2f& operator/=(float val)
    {
        v[0] /= val;
        v[1] /= val;
        return *this;
    }
    
    Vec2f operator-() const
    {
        return Vec2f(-v[0], -v[1]);
    }
    
    float v[2];
};

inline Vec2f operator+(const Vec2f &lfs, const Vec2f &rhs)
{
    return Vec2f(lfs[0] + rhs[0],
                 lfs[1] + rhs[1]);
}

inline Vec2f operator-(const Vec2f &lfs, const Vec2f &rhs)
{
    return Vec2f(lfs[0] - rhs[0],
                 lfs[1] - rhs[1]);
}

inline Vec2f operator*(const Vec2f &lfs, float rhs)
{
    return Vec2f(lfs[0] * rhs, lfs[1] * rhs);
}

inline Vec2f operator/(const Vec2f lfs, float rhs)
{
    return Vec2f(lfs[0] / rhs, lfs[1] / rhs);
}

inline float vecLength(const Vec2f &v)
{
    return std::sqrt(v[0] * v[0] + v[1] * v[1]);
}

inline Vec2f vecNormal(const Vec2f &v)
{
    const float len = 1.0f/vecLength(v);
    return Vec2f(v[0] * len, v[1] * len);
}

inline float vecDot(const Vec2f &a, const Vec2f &b)
{
    return a[0]*b[0] + a[1]*b[1];
}

inline Vec2f linearInterpolate(const Vec2f &orign, const Vec2f &dest, float u)
{
    return orign + (dest - orign) * u;
}

UD_NAMESPACE_END

#endif
