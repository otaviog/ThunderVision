#ifndef UD_VEC4_HPP
#define UD_VEC4_HPP

#include "../common.hpp"

UD_NAMESPACE_BEGIN

class Vec4f
{
public:
    Vec4f() { }

    Vec4f(float a)
    {
        v[0] = v[1] = v[2] = v[3] = a;
    }

    Vec4f(float x_, float y_, float z_, float w_)
    {
        v[0] = x_;
        v[1] = y_;
        v[2] = z_;
        v[3] = w_;
    }

    Vec4f(float a[4])
    {
        v[0] = a[0];
        v[1] = a[1];
        v[2] = a[2];
        v[3] = a[3];
    }

    void set(float x, float y, float z, float w)
    {
        v[0] = x;
        v[1] = y;
        v[2] = z;
        v[3] = w;
    }

    Vec4f operator-()
    {
        return Vec4f(-v[0], -v[1], -v[2], -v[3]);
    }

    Vec4f& operator+=(const Vec4f &rhs)
    {
        v[0] += rhs[0];
        v[1] += rhs[1];
        v[2] += rhs[2];
        v[3] += rhs[3];
        return *this;
    }

    Vec4f& operator-=(const Vec4f &rhs)
    {
        v[0] -= rhs[0];
        v[1] -= rhs[1];
        v[2] -= rhs[2];
        return *this;
    }

    Vec4f& operator*=(float rhs)
    {
        v[0] *= rhs;
        v[1] *= rhs;
        v[2] *= rhs;
        v[3] *= rhs;
        return *this;
    }

    Vec4f& operator/=(float rhs)
    {
        v[0] /= rhs;
        v[1] /= rhs;
        v[2] /= rhs;
        v[3] /= rhs;
        return *this;
    }

    float operator[](int i) const
    {
        return v[i];
    }

    float& operator[](int i)
    {
        return v[i];
    }

    float v[4];
};

inline Vec4f operator+(const Vec4f &lfs, const Vec4f &rhs)
{
    return Vec4f(lfs[0] + rhs[0], lfs[1] + rhs[1], lfs[2] + rhs[2], lfs[3] + rhs[3]);
}

inline Vec4f operator-(const Vec4f &lfs, const Vec4f &rhs)
{
    return Vec4f(lfs[0] - rhs[0], lfs[1] - rhs[1], lfs[2] - rhs[2], lfs[3] - rhs[3]);
}

inline Vec4f operator/(const Vec4f &lfs, float rhs)
{
    return Vec4f(lfs[0] / rhs, lfs[1] / rhs, lfs[2] / rhs, lfs[3] / rhs);
}

inline Vec4f operator*(const Vec4f &lfs, float rhs)
{
    return Vec4f(lfs[0] * rhs, lfs[1] * rhs, lfs[2] * rhs, lfs[3] * rhs);
}

inline Vec4f operator*(float rhs, const Vec4f &lfs)
{
    return lfs * rhs;
}

inline bool operator==(const Vec4f &lfs, const Vec4f &rhs)
{
    return lfs[0] == rhs[0] && lfs[1] == rhs[1] && lfs[2] == rhs[2];
}

inline float vecDot(const Vec4f &lfs, const Vec4f &rhs)
{
    return lfs[0] * rhs[0] + lfs[1] * rhs[1] + lfs[2] * rhs[2] + lfs[3] * rhs[3];
}

inline Vec4f vecCross(const Vec4f &lfs, const Vec4f &rhs)
{
    return Vec4f(lfs[1] * rhs[2] - lfs[2] * rhs[1],
                 lfs[2] * rhs[0] - lfs[0] * rhs[2],
                 lfs[0] * rhs[1] - lfs[1] * rhs[0],
                 1.0f);
}

inline Vec4f cosineInterpolate(const Vec4f &orign, const Vec4f &dest, float u)
{
    u = (1.0f - std::cos(u*UD_FPI))/2.0f;
    return orign + (dest - orign) * u;
}

inline Vec4f linearInterpolate(const Vec4f &orign, const Vec4f &dest, float u)
{
    return orign + (dest - orign) * u;
}


inline float vecLength(const Vec4f &v)
{
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

inline Vec4f vecNormal(const Vec4f &v)
{
    const float len = 1.0f/vecLength(v);
    return Vec4f(v[0] * len, v[1] * len, v[2] * len, v[3] * len);
}

UD_NAMESPACE_END

#endif
