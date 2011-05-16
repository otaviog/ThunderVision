#ifndef UD_MATRIX33_HPP
#define UD_MATRIX33_HPP

#include <cstring>
#include <cmath>
#include "../common.hpp"
#include "vec.hpp"
#include "quat.hpp"

UD_NAMESPACE_BEGIN


class Matrix33f
{
public:
    static const Matrix33f& Identity()
    {
        static const Matrix33f idt(
            1, 0, 0,
            0, 1, 0,
            0, 0, 1);

        return idt;
    }

    static Matrix33f RotationX(float angle)
    {
        const float cosA = std::cos(angle);
        const float sinA = std::sin(angle);

        return Matrix33f(
            1,     0,    0,
            0,  cosA, sinA,
            0, -sinA, cosA);
    }

    static Matrix33f RotationY(float angle)
    {
        const float cosA = std::cos(angle);
        const float sinA = std::sin(angle);

        return Matrix33f(
            cosA,  0, sinA,
            0,     1,    0,
            -sinA, 0, cosA
            );
    }

    static Matrix33f RotationZ(float angle)
    {
        const float cosA = std::cos(angle);
        const float sinA = std::sin(angle);

        return Matrix33f(
            cosA, sinA, 0,
            -sinA, cosA, 0,
            0,    0,  1);
    }

    static Matrix33f Rotation(float angle, float ax, float ay, float az)
    {
        //TODO
        //const float cosA = std::cos(angle);
        //const float sinA = std::sin(angle);
        return Matrix33f::Identity();
    }

    static Matrix33f Scale(float x, float y, float z)
    {
        return Matrix33f(
            x, 0, 0,
            0, y, 0,
            0, 0, z);
    }

    Matrix33f() { }

    Matrix33f(const float c[9])
    {
        memcpy(m, c, sizeof(float) * 9);
    }

    Matrix33f(const float c[3][3])
    {
        m[0] = c[0][0];
        m[3] = c[0][1];
        m[6] = c[0][2];
        m[1] = c[1][0];
        m[4] = c[1][1];
        m[7] = c[1][2];
        m[2] = c[2][0];
        m[5] = c[2][1];
        m[8] = c[2][2];
    }

    Matrix33f(const Quatf &q)
    {
        // FIXME: test
        m[0] = 1 - 2 * (q.y*q.y + q.z*q.z);
        m[1] = 2 * (q.x*q.y + q.w*q.z);
        m[2] = 2 * (q.x*q.z - q.w*q.y);

        m[3] = 2 * (q.x*q.y - q.w*q.z);
        m[4] = 1 - 2 * (q.x*q.x + q.z*q.z);
        m[5] = 2 * (q.y*q.z + q.w*q.x);

        m[6] = 2 * (q.x*q.z + q.w*q.y);
        m[7] = 2 * (q.y*q.z - q.w*q.x);
        m[8] = 1 - 2 * (q.x*q.x + q.y*q.y);
    }

    Matrix33f(float e00, float e01, float e02,
              float e10, float e11, float e12,
              float e20, float e21, float e22)
    {
        m[0] = e00;
        m[3] = e01;
        m[6] = e02;
        m[1] = e10;
        m[4] = e11;
        m[7] = e12;
        m[2] = e20;
        m[5] = e21;
        m[8] = e22;
    }

    Matrix33f(const Vec2f &col0, const Vec2f &col1)
    {
        m[0] = col0[0];
        m[3] = col1[0];
        m[1] = col0[1];
        m[4] = col1[1];
    }

    Matrix33f(const Vec3f &col0, const Vec3f &col1,
              const Vec3f &col2)
    {
        m[0] = col0[0];
        m[3] = col1[0];
        m[6] = col2[0];
        m[1] = col0[1];
        m[4] = col1[1];
        m[7] = col2[1];
        m[2] = col0[2];
        m[5] = col1[2];
        m[8] = col2[2];
    }

    inline Matrix33f& operator+=(const Matrix33f &rhs);
    inline Matrix33f& operator-=(const Matrix33f &rhs);
    inline Matrix33f& operator*=(const Matrix33f &rhs);
    inline Matrix33f& operator*=(const float rhs);
    inline Matrix33f& operator/=(const float rhs);

    float& operator()(int i, int j)
    {
        return m[i + j*3];
    }

    const float& operator()(int i, int j) const
    {
        return m[i + j*3];
    }

    float& operator[](int i)
    {
        return m[i];
    }

    const float& operator[](int i) const
    {
        return m[i];
    }

    Vec3f column(int c) const
    {
        switch (c)
        {
        case 0:
            return Vec3f(m[0], m[1], m[2]);
        case 1:
            return Vec3f(m[3], m[4], m[5]);
        case 2:
        default:
            return Vec3f(m[6], m[7], m[8]);
        }
    }

    float m[9];
};

inline Matrix33f operator+(const Matrix33f &lfs, const Matrix33f &rhs)
{
    return Matrix33f(lfs[0] + rhs[0], lfs[3] + rhs[3], lfs[6] + rhs[6],
                     lfs[1] + rhs[1], lfs[4] + rhs[4], lfs[7] + rhs[7],
                     lfs[2] + rhs[2], lfs[5] + rhs[5], lfs[8] + rhs[8]);
}

inline Matrix33f operator-(const Matrix33f &lfs, const Matrix33f &rhs)
{
    return Matrix33f(lfs[0] - rhs[0], lfs[3] - rhs[3], lfs[6] - rhs[6],
                     lfs[1] - rhs[1], lfs[4] - rhs[4], lfs[7] - rhs[7],
                     lfs[2] - rhs[2], lfs[5] - rhs[5], lfs[8] - rhs[8]);
}

inline Matrix33f operator/(const Matrix33f &lfs, float rhs)
{
    return Matrix33f(lfs[0] / rhs, lfs[3] / rhs, lfs[6] / rhs,
                     lfs[1] / rhs, lfs[4] / rhs, lfs[7] / rhs,
                     lfs[2] / rhs, lfs[5] / rhs, lfs[8] / rhs);
}

inline Matrix33f operator*(const Matrix33f &lfs, float rhs)
{
    return Matrix33f(lfs[0] * rhs, lfs[3] * rhs, lfs[6] * rhs,
                     lfs[1] * rhs, lfs[4] * rhs, lfs[7] * rhs,
                     lfs[2] * rhs, lfs[5] * rhs, lfs[8] * rhs);
}

inline Vec3f operator*(const Vec3f &lfs, const Matrix33f &rhs)
{
    return Vec3f(lfs[0] * rhs[0] + lfs[1] * rhs[1] + lfs[2] * rhs[2],
                 lfs[0] * rhs[3] + lfs[1] * rhs[4] + lfs[2] * rhs[5],
                 lfs[0] * rhs[6] + lfs[1] * rhs[7] + lfs[2] * rhs[8]);
}

inline Vec2f operator*(const Vec2f &lfs, const Matrix33f &rhs)
{
    return Vec2f(lfs[0]*rhs[0] + lfs[1]*rhs[1] + rhs[2],
                 lfs[0]*rhs[3] + lfs[1]*rhs[4] + rhs[5]);
}

inline Matrix33f operator*(const Matrix33f &lfs, const Matrix33f &rhs)
{
    return Matrix33f(lfs[0] * rhs[0] + lfs[3] * rhs[1] + lfs[6] * rhs[2],
                     lfs[0] * rhs[3] + lfs[3] * rhs[4] + lfs[6] * rhs[5],
                     lfs[0] * rhs[6] + lfs[3] * rhs[7] + lfs[6] * rhs[8],

                     lfs[1] * rhs[0] + lfs[4] * rhs[1] + lfs[7] * rhs[2],
                     lfs[1] * rhs[3] + lfs[4] * rhs[4] + lfs[7] * rhs[5],
                     lfs[1] * rhs[6] + lfs[4] * rhs[7] + lfs[7] * rhs[8],

                     lfs[2] * rhs[0] + lfs[5] * rhs[1] + lfs[8] * rhs[2],
                     lfs[2] * rhs[3] + lfs[5] * rhs[4] + lfs[8] * rhs[5],
                     lfs[2] * rhs[6] + lfs[5] * rhs[7] + lfs[8] * rhs[8]);
}

Matrix33f& Matrix33f::operator+=(const Matrix33f &rhs)
{
    *this = *this + rhs;
    return *this;
}

Matrix33f& Matrix33f::operator-=(const Matrix33f &rhs)
{
    *this = *this - rhs;
    return *this;
}

Matrix33f& Matrix33f::operator*=(const Matrix33f &rhs)
{
    *this = *this * rhs;
    return *this;
}

Matrix33f& Matrix33f::operator*=(const float rhs)
{
    *this = *this * rhs;
    return *this;
}

Matrix33f& Matrix33f::operator/=(const float rhs)
{
    *this = *this / rhs;
    return *this;
}

inline Matrix33f matrixTranspose(const Matrix33f &m)
{
    return Matrix33f(m[0], m[1], m[2],
                     m[3], m[4], m[5],
                     m[6], m[7], m[8]);
}

inline float matrixDeterminant(const Matrix33f &lfs)
{
    return lfs[0] * (lfs[4] * lfs[8] - lfs[5] * lfs[7])
        - lfs[3] * (lfs[1] * lfs[8] - lfs[2] * lfs[7])
        + lfs[6] * (lfs[1] * lfs[5] - lfs[2] * lfs[4]);
}

inline Matrix33f matrixCofactor(const Matrix33f &m)
{

    return Matrix33f(m[4] * m[8] - m[5] * m[7],
                     -(m[1] * m[8] - m[2] * m[7]),
                     m[1] * m[5] - m[2] * m[4],

                     -(m[3] * m[8] - m[5] * m[6]),
                     m[0] * m[8] - m[2] * m[6],
                     -(m[0] * m[5] - m[2] * m[3]),

                     m[3] * m[7] - m[4] * m[6],
                     -(m[0] * m[7] - m[1] * m[6]), //-(m[0] * m[8] - m[1] * m[6]),
                     m[0] * m[4] - m[1] * m[3]);
}

inline bool matrixInverse(const Matrix33f &m, Matrix33f &ctm)
{
    bool r;
    const float det = matrixDeterminant(m);

    if ( det != 0 )
    {
        ctm = matrixCofactor(matrixTranspose(m)) / det;
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
