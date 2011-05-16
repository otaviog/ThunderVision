#ifndef UD_MATRIX22_HPP
#define UD_MATRIX22_HPP

#include <cmath>
#include <cstring>
#include "../common.hpp"
#include "vec.hpp"

UD_NAMESPACE_BEGIN


class Matrix22f
{
public:
    static const Matrix22f& Identity()
    {
        static const Matrix22f idt(
            1, 0,
            0, 1);

        return idt;
    }

    static Matrix22f Rotate(float angle)
    {
        const float cosA = std::cos(angle);
        const float sinA = std::sin(angle);

        return Matrix22f(
            cosA, sinA,
            -sinA, cosA);
    }

    static Matrix22f Scale(float x, float y)
    {
        return Matrix22f(
            x, 0,
            0, y);
    }

    Matrix22f() { }

    Matrix22f(const float c[4])
    {
        memcpy(m, c, sizeof(float) * 4);
    }

    Matrix22f(const float c[2][2])
    {
        m[0] = c[0][0];
        m[2] = c[0][1];
        m[1] = c[1][0];
        m[3] = c[1][1];
    }

    Matrix22f(float e00, float e01,
              float e10, float e11)
    {
        m[0] = e00;
        m[2] = e01;
        m[1] = e10;
        m[3] = e11;
    }

    Matrix22f(const Vec2f &col0, const Vec2f &col1)
    {
        m[0] = col0[0];
        m[2] = col1[0];
        m[1] = col0[1];
        m[3] = col1[1];
    }

    inline Matrix22f& operator+=(const Matrix22f &rhs);

    inline Matrix22f& operator-=(const Matrix22f &rhs);

    inline Matrix22f& operator*=(const Matrix22f &rhs);

    inline Matrix22f& operator*=(float rhs);

    const float& operator[](int i) const
    {
        return m[i];
    }

    float& operator[](int i)
    {
        return m[i];
    }

    const float& operator()(int r, int c) const
    {
        return m[r + c*2];
    }

    float& operator()(int r, int c)
    {
        return m[r + c*2];
    }

    float m[4];
};

inline Matrix22f operator+(const Matrix22f &lfs, const Matrix22f &rhs)
{
    return Matrix22f(
        lfs[0] + rhs[0], lfs[2] + rhs[2],
        lfs[1] + rhs[1], lfs[3] + rhs[3]);
}

inline Matrix22f operator-(const Matrix22f &lfs, const Matrix22f &rhs)
{
    return Matrix22f(
        lfs[0] - rhs[0], lfs[2] - rhs[2],
        lfs[1] - rhs[1], lfs[3] - rhs[3]);
}

inline Matrix22f operator*(const Matrix22f &lfs, float rhs)
{
    return Matrix22f(
        lfs[0] * rhs, lfs[2] * rhs,
        lfs[1] * rhs, lfs[3] * rhs);
}

inline Vec2f operator*(const Vec2f &lfs, const Matrix22f &rhs)
{
    return Vec2f(
        lfs[0] * rhs[0] + lfs[1] * rhs[1],
        lfs[0] * rhs[2] + lfs[1] * rhs[3]);
}

inline Matrix22f operator*(const Matrix22f &lfs, const Matrix22f &rhs)
{
    return Matrix22f(
        lfs[0] * rhs[0] + lfs[2] * rhs[1],
        lfs[0] * rhs[2] + lfs[2] * rhs[3],

        lfs[1] * rhs[0] + lfs[3] * rhs[1],
        lfs[1] * rhs[2] + lfs[3] * rhs[3]);
}

Matrix22f& Matrix22f::operator+=(const Matrix22f &rhs)
{
    *this = *this + rhs;
    return *this;
}

Matrix22f& Matrix22f::operator-=(const Matrix22f &rhs)
{
    *this = *this - rhs;
    return *this;
}

Matrix22f& Matrix22f::operator*=(const Matrix22f &rhs)
{
    *this = *this * rhs;
    return *this;
}

Matrix22f& Matrix22f::operator*=(float rhs)
{
    *this = *this * rhs;
    return *this;
}

inline float matrixDeterminant(const Matrix22f &lfs)
{
    return lfs[0] * lfs[3] - lfs[1] * lfs[2];
}

inline bool matrixInverse(const Matrix22f &lfs, Matrix22f &rhs)
{
    const float d = matrixDeterminant(lfs);
    bool r;

    if ( d != 0 )
    {
        rhs[0] = lfs[3] / d;
        rhs[1] = -lfs[1] / d;
        rhs[2] = -lfs[2] / d;
        rhs[3] = lfs[0] / d;
        r = true;
    }
    else
    {
        r = false;
    }

    return r;
}

UD_NAMESPACE_END

#endif
