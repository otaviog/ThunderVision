#ifndef UD_MATRIX44_HPP
#define UD_MATRIX44_HPP
#include <ostream>
#include <cmath>
#include <cstring>
#include "../common.hpp"
#include "vec.hpp"
#include "quat.hpp"

UD_NAMESPACE_BEGIN

/**
 * 4x4 Matrix implementation
 */

class Matrix44f
{
public:
    static const Matrix44f& Identity()
    {
        static const Matrix44f idt(
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1);

        return idt;
    }

    static Matrix44f Rotation(float angle, float nx, float ny, float nz)
    {
        float cosA = std::cos(angle);
        float sinA = std::sin(angle);

        return Matrix44f(nx*nx*(1-cosA) + cosA,
                         nx*ny*(1-cosA) - nz*sinA,
                         nx*nz*(1-cosA) + ny*sinA,
                         0,
                         nx*ny*(1-cosA) + nx*sinA,
                         ny*ny*(1-cosA) + cosA,
                         ny*nz*(1-cosA) - nx*sinA,
                         0,
                         nx*nz*(1-cosA) - ny*sinA,
                         ny*nz*(1-cosA) + nx*sinA,
                         nz*nz*(1-cosA) + cosA,
                         0,
                         0, 0, 0, 1);
    }

    static Matrix44f Rotation(float angle, const Vec3f &v)
    {
        return Rotation(angle, v[0], v[1], v[2]);
    }

    static Matrix44f RotationX(float angle)
    {
        const float cosA = std::cos(angle);
        const float sinA = std::sin(angle);

        return Matrix44f(
            1,     0,    0, 0,
            0,  cosA, sinA, 0,
            0, -sinA, cosA, 0,
            0,     0,    0, 1);
    }

    static Matrix44f RotationY(float angle)
    {
        const float cosA = std::cos(angle);
        const float sinA = std::sin(angle);

        return Matrix44f(
            cosA, 0, sinA, 0,
            0, 1,    0, 0,
            -sinA, 0, cosA, 0,
            0, 0,    0, 1);
    }

    static Matrix44f RotationZ(float angle)
    {
        const float cosA = std::cos(angle);
        const float sinA = std::sin(angle);

        return Matrix44f(
            cosA,  sinA, 0, 0,
            -sinA, cosA, 0, 0,
            0,    0, 1, 0,
            0,    0, 0, 1);
    }

    static Matrix44f Translation(float x, float y, float z)
    {
        return Matrix44f(
            1, 0, 0, x,
            0, 1, 0, y,
            0, 0, 1, z,
            0, 0, 0, 1);
    }

    static Matrix44f Translation(const Vec3f &v)
    {
        return Translation(v[0], v[1], v[2]);
    }

    static Matrix44f Scale(float x, float y, float z)
    {
        return Matrix44f(
            x, 0, 0, 0,
            0, y, 0, 0,
            0, 0, z, 0,
            0, 0, 0, 1.0f);
    }

    static Matrix44f Scale(const Vec3f &v)
    {
        return Scale(v[0], v[1], v[2]);
    }

    static inline Matrix44f Crop(float scaleX, float scaleY, float scaleZ,
                          float offsetX, float offsetY, float offsetZ)

    {
        return Matrix44f(scaleX, 0.0f, 0.0f, offsetX,
                         0.0f, scaleY, 0.0f, offsetY,
                         0.0f, 0.0f, scaleZ, offsetZ,
                         0.0f, 0.0f, 0.0f, 1.0f);
    }

    static inline Matrix44f TextureProjectionCrop()
    {
        static Matrix44f crop = Crop(0.5f, 0.5f, 0.5f,
                                     0.5f, 0.5f, 0.5f);
        return crop;
    }

    /**
     * Atetion the default constructor matrix data is not pre initialized.
     */
    Matrix44f() { }

    Matrix44f(const Matrix44f &copy)
    {
        memcpy(m, copy.m, sizeof(float) * 16);
    }

    inline Matrix44f(const Vec3f trans, const Quatf &orientation);

    inline Matrix44f(const Quatf &orientation, const Vec3f trans);

    /**
     * OpenGL style matrix constructor
     */
    inline explicit Matrix44f(float gl_mtx[16]);

    /**
     * Minucious style constructor
     */
    inline Matrix44f(float e00, float e01, float e02, float e03,
              float e10, float e11, float e12, float e13,
              float e20, float e21, float e22, float e23,
              float e30, float e31, float e32, float e33);

    Matrix44f(const Vec2f &col0, const Vec2f &col1)
    {
        m[0] = col0[0];
        m[2] = 0;
        m[3] = 0;
        m[4] = col1[0];
        m[1] = col0[1];
        m[5] = col1[1];
        m[6] = m[7] = m[8] = m[9]
            = m[10] = m[11] = m[12]
            = m[13] = m[14] = m[15]
            = 0;
    }

    Matrix44f(const Vec3f &col0, const Vec3f &col1,
              const Vec3f &col2)
    {
        m[0] = col0[0];
        m[4] = col1[0];
        m[8] = col2[0];
        m[1] = col0[1];
        m[5] = col1[1];
        m[9] = col2[1];
        m[2] = col0[2];
        m[6] = col1[2];
        m[10] = col2[2];
    }

    Matrix44f(const Vec4f &col0, const Vec4f &col1,
              const Vec4f &col2, const Vec4f &col3)
    {
        m[0] = col0[0];
        m[4] = col1[0];
        m[8] = col2[0];
        m[12] = col3[0];
        m[1] = col0[1];
        m[5] = col1[1];
        m[9] = col2[1];
        m[13] = col3[1];
        m[2] = col0[2];
        m[6] = col1[2];
        m[10] = col2[2];
        m[14] = col3[2];
        m[3] = col0[3];
        m[7] = col1[3];
        m[11] = col2[3];
        m[15] = col3[3];
    }

    /**
     * C++ style matrix
     */
    inline explicit Matrix44f(float mtx[4][4]);

    inline explicit Matrix44f(const Quatf &quat);

    inline Vec3f multRotPart(const Vec3f &lfs) const
    {
        return Vec3f(lfs[0] * m[0] + lfs[1] * m[4] + lfs[2] * m[8],
                     lfs[0] * m[1] + lfs[1] * m[5] + lfs[2] * m[9],
                     lfs[0] * m[2] + lfs[1] * m[6] + lfs[2] * m[10]);
    }

    void loadIdentity()
    {
        m[0] = 1;
        m[1] = 0;
        m[2] = 0;
        m[3] = 0;

        m[4] = 0;
        m[5] = 1;
        m[6] = 0;
        m[7] = 0;

        m[8] = 0;
        m[9] = 0;
        m[10] = 1;
        m[11] = 0;

        m[12] = 0;
        m[13] = 0;
        m[14] = 0;
        m[15] = 1;

    }

    float operator[](int index) const
    {
        return m[index];
    }

    float& operator[](int index)
    {
        return m[index];
    }

    float& at(int r, int c)
    {
        return m[r + c*4];
    }
    float at(int r, int c) const
    {
        return m[r + c*4];
    }

    float& operator()(int r, int c)
    {
        return at(r, c);
    }

    float operator()(int r, int c) const
    {
        return at(r, c);
    }

    inline Matrix44f& operator+=(const Matrix44f &rhs);

    inline Matrix44f& operator-=(const Matrix44f &rhs);

    inline Matrix44f& operator*=(float rhs);

    inline Matrix44f& operator*=(const Matrix44f &rhs);

    inline Matrix44f& operator/=(float rhs);

    const float* column(int col) const
    {
        return &m[col*4];
    }

    Vec4f columnV(int col) const
    {
        switch (col)
        {
        case 0:
            return Vec4f(m[0], m[1], m[2], m[3]);
            break;
        case 1:
            return Vec4f(m[4], m[5], m[6], m[7]);
            break;
        case 2:
            return Vec4f(m[8], m[9], m[10], m[11]);
            break;
        case 3:
        default:
            return Vec4f(m[12], m[13], m[14], m[15]);
            break;
        }
    }

    float m[16];
};

Matrix44f::Matrix44f(float gl_mtx[16])
{
    m[0] = gl_mtx[0];
    m[4] = gl_mtx[4];
    m[8] = gl_mtx[8];
    m[12] = gl_mtx[12];

    m[1] = gl_mtx[1];
    m[5] = gl_mtx[5];
    m[9] = gl_mtx[9];
    m[13] = gl_mtx[13];

    m[2] = gl_mtx[2];
    m[6] = gl_mtx[6];
    m[10] =gl_mtx[10];
    m[14] = gl_mtx[14];

    m[3] = gl_mtx[3];
    m[7] = gl_mtx[7];
    m[11] = gl_mtx[11];
    m[15] = gl_mtx[15];
}

Matrix44f::Matrix44f(float e00, float e01, float e02, float e03,
                     float e10, float e11, float e12, float e13,
                     float e20, float e21, float e22, float e23,
                     float e30, float e31, float e32, float e33)
{
    m[0] = e00;
    m[4] = e01;
    m[8] = e02;
    m[12] = e03;
    m[1] = e10;
    m[5] = e11;
    m[9] = e12;
    m[13] = e13;
    m[2] = e20;
    m[6] = e21;
    m[10] = e22;
    m[14] = e23;
    m[3] = e30;
    m[7] = e31;
    m[11] = e32;
    m[15] = e33;
}

Matrix44f::Matrix44f(float mtx[4][4])
{
    m[0] = mtx[0][0];
    m[4] = mtx[0][1];
    m[8] = mtx[0][2];
    m[12] = mtx[0][3];
    m[1] = mtx[1][0];
    m[5] = mtx[1][1];
    m[9] = mtx[1][2];
    m[13] = mtx[1][3];
    m[2] = mtx[2][0];
    m[2] = mtx[2][1];
    m[10] = mtx[2][2];
    m[14] = mtx[2][3];
    m[3] = mtx[3][0];
    m[7] = mtx[3][1];
    m[11] = mtx[3][2];
    m[15] = mtx[3][3];
}

/**
 * Extracted from 3D Math Primer for Graphics and Game Development.
 */
Matrix44f::Matrix44f(const Quatf &q)
{
    m[0] = 1.0f - 2.0f*(q.y*q.y + q.z*q.z);
    m[1] = 2.0f*(q.x*q.y + q.w*q.z);
    m[2] = 2.0f*(q.x*q.z - q.w*q.y);
    m[3] = 0.0f;

    m[4] = 2.0f*(q.x*q.y - q.w*q.z);
    m[5] = 1.0f - 2.0f * (q.x*q.x + q.z*q.z);
    m[6] = 2.0f * (q.y*q.z + q.w*q.x);
    m[7] = 0.0f;

    m[8] = 2.0f * (q.x*q.z + q.w*q.y);
    m[9] = 2.0f * (q.y*q.z - q.w*q.x);
    m[10] = 1.0f - 2.0f * (q.x*q.x + q.y*q.y);
    m[11] = 0.0f;

    m[12] = 0.0f;
    m[13] = 0.0f;
    m[14] = 0.0f;
    m[15] = 1.0f;
}

inline Matrix44f operator+(const Matrix44f &lfs, const Matrix44f &rhs)
{
    return Matrix44f(
        lfs.m[0] + rhs.m[0], lfs.m[4] + rhs.m[4], lfs.m[8] + rhs.m[8], lfs.m[12] + rhs.m[12],
        lfs.m[1] + rhs.m[1], lfs.m[5] + rhs.m[5], lfs.m[9] + rhs.m[9], lfs.m[13] + rhs.m[13],
        lfs.m[2] + rhs.m[2], lfs.m[6] + rhs.m[6], lfs.m[10] + rhs.m[10], lfs.m[14] + rhs.m[14],
        lfs.m[3] + rhs.m[3], lfs.m[7] + rhs.m[7], lfs.m[11] + rhs.m[11], lfs.m[15] + rhs.m[15]);
}

inline Matrix44f operator-(const Matrix44f &lfs, const Matrix44f &rhs)
{
    return Matrix44f(
        lfs.m[0] - rhs.m[0], lfs.m[4] - rhs.m[4], lfs.m[8] - rhs.m[8], lfs.m[12] - rhs.m[12],
        lfs.m[1] - rhs.m[1], lfs.m[5] - rhs.m[5], lfs.m[9] - rhs.m[9], lfs.m[13] - rhs.m[13],
        lfs.m[2] - rhs.m[2], lfs.m[6] - rhs.m[6], lfs.m[10] - rhs.m[10], lfs.m[14] - rhs.m[14],
        lfs.m[3] - rhs.m[3], lfs.m[7] - rhs.m[7], lfs.m[11] - rhs.m[11], lfs.m[15] - rhs.m[15]);
}

inline Matrix44f operator*(const Matrix44f &lfs, float rhs)
{
    return Matrix44f(
        lfs.m[0] * rhs, lfs.m[4] * rhs, lfs.m[8] * rhs, lfs.m[12] * rhs,
        lfs.m[1] * rhs, lfs.m[5] * rhs, lfs.m[9] * rhs, lfs.m[13] * rhs,
        lfs.m[2] * rhs, lfs.m[6] * rhs, lfs.m[10] * rhs, lfs.m[14] * rhs,
        lfs.m[3] * rhs, lfs.m[7] * rhs, lfs.m[11] * rhs, lfs.m[15] * rhs);
}

inline Vec3f operator*(const Vec3f &lfs, const Matrix44f &rhs)
{
    return Vec3f(
        lfs[0] * rhs.m[0] + lfs[1] * rhs.m[4] + lfs[2] * rhs.m[8] + rhs.m[12],
        lfs[0] * rhs.m[1] + lfs[1] * rhs.m[5] + lfs[2] * rhs.m[9] + rhs.m[13],
        lfs[0] * rhs.m[2] + lfs[1] * rhs.m[6] + lfs[2] * rhs.m[10] + rhs.m[14]);
}

inline Vec4f operator*(const Vec4f &lfs, const Matrix44f &rhs)
{
    return Vec4f(
        lfs[0] * rhs.m[0] + lfs[1] * rhs.m[4] + lfs[2] * rhs.m[8] + lfs[3] * rhs.m[12],
        lfs[0] * rhs.m[1] + lfs[1] * rhs.m[5] + lfs[2] * rhs.m[9] + lfs[3] * rhs.m[13],
        lfs[0] * rhs.m[2] + lfs[1] * rhs.m[6] + lfs[2] * rhs.m[10] + lfs[3] * rhs.m[14],
        lfs[0] * rhs.m[3] + lfs[1] * rhs.m[7] + lfs[2] * rhs.m[11] + lfs[3] * rhs.m[15]);
}

inline Matrix44f operator*(const Matrix44f &lfs, const Matrix44f &rhs)
{
    return Matrix44f(
        lfs.m[0] * rhs.m[0] + lfs.m[4] * rhs.m[1]
        + lfs.m[8] * rhs.m[2] + lfs.m[12] * rhs.m[3],
        lfs.m[0] * rhs.m[4] + lfs.m[4] * rhs.m[5]
        + lfs.m[8] * rhs.m[6] + lfs.m[12] * rhs.m[7],
        lfs.m[0] * rhs.m[8] + lfs.m[4] * rhs.m[9]
        + lfs.m[8] * rhs.m[10] + lfs.m[12] * rhs.m[11],
        lfs.m[0] * rhs.m[12] + lfs.m[4] * rhs.m[13]
        + lfs.m[8] * rhs.m[14] + lfs.m[12] * rhs.m[15],

        lfs.m[1] * rhs.m[0] + lfs.m[5] * rhs.m[1]
        + lfs.m[9] * rhs.m[2] + lfs.m[13] * rhs.m[3],
        lfs.m[1] * rhs.m[4] + lfs.m[5] * rhs.m[5]
        +lfs.m[9] * rhs.m[6] + lfs.m[13] * rhs.m[7],
        lfs.m[1] * rhs.m[8] + lfs.m[5] * rhs.m[9]
        + lfs.m[9] * rhs.m[10] + lfs.m[13] * rhs.m[11],
        lfs.m[1] * rhs.m[12] + lfs.m[5] * rhs.m[13]
        + lfs.m[9] * rhs.m[14] + lfs.m[13] * rhs.m[15],

        lfs.m[2] * rhs.m[0] + lfs.m[6] * rhs.m[1]
        + lfs.m[10] * rhs.m[2] + lfs.m[14] * rhs.m[3],
        lfs.m[2] * rhs.m[4] + lfs.m[6] * rhs.m[5]
        + lfs.m[10] * rhs.m[6] + lfs.m[14] * rhs.m[7],
        lfs.m[2] * rhs.m[8] + lfs.m[6] * rhs.m[9]
        + lfs.m[10] * rhs.m[10] + lfs.m[14] * rhs.m[11],
        lfs.m[2] * rhs.m[12] + lfs.m[6] * rhs.m[13]
        + lfs.m[10] * rhs.m[14] + lfs.m[14] * rhs.m[15],

        lfs.m[3] * rhs.m[0] + lfs.m[7] * rhs.m[1]
        + lfs.m[11] * rhs.m[2] + lfs.m[15] * rhs.m[3],
        lfs.m[3] * rhs.m[4] + lfs.m[7] * rhs.m[5]
        + lfs.m[11] * rhs.m[6] + lfs.m[15] * rhs.m[7],
        lfs.m[3] * rhs.m[8] + lfs.m[7] * rhs.m[9]
        + lfs.m[11] * rhs.m[10] + lfs.m[15] * rhs.m[11],
        lfs.m[3] * rhs.m[12] + lfs.m[7] * rhs.m[13]
        + lfs.m[11] * rhs.m[14] + lfs.m[15] * rhs.m[15]);
}

inline Matrix44f operator/(const Matrix44f &lfs, float rhs)
{
    return lfs * (1.0f/rhs);
}

Matrix44f::Matrix44f(const Quatf &orientation, const Vec3f trans)
{
    *this = Matrix44f(orientation) * Translation(trans);
}

Matrix44f::Matrix44f(const Vec3f trans, const Quatf &orientation)
{
    *this = Matrix44f::Translation(trans) * Matrix44f(orientation);
}

Matrix44f& Matrix44f::operator+=(const Matrix44f &rhs)
{
    *this = *this + rhs;
    return *this;
}

Matrix44f& Matrix44f::operator-=(const Matrix44f &rhs)
{
    *this = *this - rhs;
    return *this;
}

Matrix44f& Matrix44f::operator*=(float rhs)
{
    *this = *this * rhs;
    return *this;
}

Matrix44f& Matrix44f::operator*=(const Matrix44f &rhs)
{
    *this = *this * rhs;
    return *this;
}

Matrix44f& Matrix44f::operator/=(float rhs)
{
    *this = *this / rhs;
    return *this;
}

inline Matrix44f matrixTranspose(const Matrix44f &m)
{
    return Matrix44f(m[0], m[1], m[2], m[3],
                     m[4], m[5], m[6], m[7],
                     m[8], m[9], m[10], m[11],
                     m[12], m[13], m[14], m[15]);
}

inline float matrixDeterminant(const Matrix44f &m)
{
    float a1, a2, a3, a4;

    //m[5] m[9] m[13]
    //m[6] m[10] m[14]
    //m[7] m[11] m[15]
    a1 = m[0] * (m[5] * (m[10] * m[15] - m[14] * m[11])
                 - m[9] * (m[6] * m[15] - m[14] * m[7])
                 + m[13] * ( m[6] * m[11] - m[10] * m[7]));

    //m[1] m[9] m[13]
    //m[2] m[10] m[14]
    //m[3] m[11] m[15]
    a2 = m[4] * (m[1] * (m[10] * m[15] - m[14] * m[11])
                 - m[9] * (m[2] * m[15] - m[14] * m[3])
                 + m[13] * (m[2] * m[11] - m[10] * m[3]));

    //m[1] m[5] m[13]
    //m[2] m[6] m[14]
    //m[3] m[7] m[15]
    a3 = m[8] * (m[1] * (m[6] * m[15] - m[14] * m[7])
                 - m[5] * (m[2] * m[15] - m[14] * m[3])
                 + m[13] * (m[2] * m[7] - m[6] * m[3]));

    //m[1] m[5] m[9]
    //m[2] m[6] m[10]
    //m[3] m[7] m[11]
    a4 = m[12] * (m[1] * (m[6] * m[11] - m[10] * m[7])
                  - m[5] * (m[2] * m[11] - m[10] * m[3])
                  + m[9] * (m[2] * m[7] - m[6] * m[3]));

    return a1 - a2 + a3 - a4;
}

inline Matrix44f matrixCofactor(const Matrix44f &mtx)
{
    return Matrix44f(
        mtx[5] * (mtx[10] * mtx[15] - mtx[11] * mtx[14])
        - mtx[9] * (mtx[6] * mtx[15] - mtx[7] * mtx[14])
        + mtx[13] * (mtx[6] * mtx[11] - mtx[7] * mtx[10]),

        // m[1] = 0; m[9] = 0; m[13] = 0;
        // m[2] = 0; m[10] = 1; m[14] = 0;
        // m[3] = 0; m[11] = 0; m[15] = 1;
        -(mtx[1] * (mtx[10] * mtx[15] - mtx[11] * mtx[14])
          - mtx[9] * (mtx[2] * mtx[15] - mtx[3] * mtx[14])
          + mtx[13] * (mtx[2] * mtx[11] - mtx[3] * mtx[10])),

        // m[1] = 0; m[5] = 1; m[13] = 0;
        // m[2] = 0; m[6] = 0; m[14] = 0;
        // m[3] = 0; m[7] = 0; m[15] = 1;
        mtx[1] * (mtx[6] * mtx[15] - mtx[7] * mtx[14])
        - mtx[5] * (mtx[2] * mtx[15] - mtx[3] * mtx[14])
        + mtx[13] * (mtx[2] * mtx[7] - mtx[3] * mtx[6]),

        // m[1] = 0; m[5] = 1; m[9] = 0;
        // m[2] = 0; m[6] = 0; m[10] = 1;
        // m[3] = 0; m[7] = 0; m[11] = 0;
        -(mtx[1] * (mtx[6] * mtx[11] - mtx[7] * mtx[10])
          - mtx[5] * (mtx[2] * mtx[11] - mtx[3] * mtx[10])
          + mtx[9] * (mtx[2] * mtx[7] - mtx[3] * mtx[6])),

        // m[4] = 0; m[8] = 0; m[12] = 0;
        // m[6] = 0; m[10] = 1; m[14] = 0;
        // m[7] = 0; m[11] = 0; m[15] = 1;
        -(mtx[4] * (mtx[10] * mtx[15] - mtx[11] * mtx[14])
          - mtx[8] * (mtx[6] * mtx[15] - mtx[7] * mtx[14])
          + mtx[12] * (mtx[6] * mtx[11] - mtx[7] * mtx[10])),

        // m[0] = 1; m[8] = 0; m[12] = 0;
        // m[2] = 0; m[10] = 1; m[14] = 0;
        // m[3] = 0; m[11] = 0; m[15] = 1;
        mtx[0] * (mtx[10] * mtx[15] - mtx[11] * mtx[14])
        - mtx[8] * (mtx[2] * mtx[15] - mtx[3] * mtx[14])
        + mtx[12] * (mtx[2] * mtx[11] - mtx[3] * mtx[10]),

        // m[0] = 1; m[4] = 0; m[12] = 0;
        // m[2] = 0; m[6] = 0; m[14] = 0;
        // m[3] = 0; m[7] = 0; m[15] = 1;
        -(mtx[0] * (mtx[6] * mtx[15] - mtx[7] * mtx[14])
          - mtx[4] * (mtx[2] * mtx[15] - mtx[3] * mtx[14])
          + mtx[12] * (mtx[2] * mtx[7] - mtx[3] * mtx[6])),

        //m[0] = 1; m[4] = 0; m[8] = 0;
        //m[2] = 0; m[6] = 0; m[10] = 1;
        //m[3] = 0; m[7] = 0; m[11] = 0;
        mtx[0] * (mtx[6] * mtx[11] - mtx[7] * mtx[10])
        - mtx[4] * (mtx[2] * mtx[11] - mtx[3] * mtx[10])
        + mtx[8] * (mtx[2] * mtx[7] - mtx[3] * mtx[6]),

        //m[4] = 0; m[8] = 0; m[12] = 0;
        //m[5] = 1; m[9] = 0; m[13] = 0;
        //m[7] = 0; m[11] = 0; m[15] = 1;
        mtx[4] * (mtx[9] * mtx[15] - mtx[11] * mtx[13])
        - mtx[8] * (mtx[5] * mtx[15] - mtx[7] * mtx[13])
        + mtx[12] * (mtx[5] * mtx[11] - mtx[7] * mtx[9]),

        //m[0] = 1; m[8] = 0; m[12] = 0;
        //m[1] = 0; m[9] = 0; m[13] = 0;
        //m[3] = 0; m[11] = 0; m[15] = 1;
        -(mtx[0] * (mtx[9] * mtx[15] - mtx[11] * mtx[13])
          - mtx[8] * (mtx[1] * mtx[15] - mtx[3] * mtx[13])
          + mtx[12] * (mtx[1] * mtx[11] - mtx[3] * mtx[9])),

        //m[0] = 1; m[4] = 0; m[12] = 0;
        //m[1] = 0; m[5] = 1; m[13] = 0;
        //m[3] = 0; m[7] = 0; m[15] = 1;
        mtx[0] * (mtx[5] * mtx[15] - mtx[7] * mtx[13])
        - mtx[4] * (mtx[1] * mtx[15] - mtx[3] * mtx[13])
        + mtx[12] * (mtx[1] * mtx[7] - mtx[3] * mtx[5]),

        // m[0] = 1; m[4] = 0; m[8] = 0;
        // m[1] = 0; m[5] = 1; m[9] = 0;
        // m[3] = 0; m[7] = 0; m[11] = 0;
        -(mtx[0] * (mtx[5] * mtx[11] - mtx[7] * mtx[9])
          - mtx[4] * (mtx[1] * mtx[11] - mtx[3] * mtx[9])
          + mtx[8] * (mtx[1] * mtx[7] - mtx[3] * mtx[5])),

        // m[4] = 0; m[8] = 0; m[12] = 0;
        // m[5] = 1; m[9] = 0; m[13] = 0;
        // m[6] = 0; m[10] = 1; m[14] = 0;
        -(mtx[4] * (mtx[9] * mtx[14] - mtx[10] * mtx[13])
          - mtx[8] * (mtx[5] * mtx[14] - mtx[6] * mtx[13])
          + mtx[12] * (mtx[5] * mtx[10] - mtx[6] * mtx[9])),

        // m[0] = 1; m[8] = 0; m[12] = 0;
        // m[1] = 0; m[9] = 0; m[13] = 0;
        // m[2] = 0; m[10] = 1; m[14] = 0;
        mtx[0] * (mtx[9] * mtx[14] - mtx[10] * mtx[13])
        - mtx[8] * (mtx[1] * mtx[14] - mtx[2] * mtx[13])
        + mtx[12] * (mtx[1] * mtx[10] - mtx[2] * mtx[9]),

        // m[0] = 1; m[4] = 0; m[12] = 0;
        // m[1] = 0; m[5] = 1; m[13] = 0;
        // m[2] = 0; m[6] = 0; m[14] = 0;
        -(mtx[0] * (mtx[5] * mtx[14] - mtx[6] * mtx[13])
          - mtx[4] * (mtx[1] * mtx[14] - mtx[2] * mtx[13])
          + mtx[12] * (mtx[1] * mtx[6] - mtx[2] * mtx[5])),

        //m[0] = 1; m[4] = 0; m[8] = 0;
        //m[1] = 0; m[5] = 1; m[9] = 0;
        //m[2] = 0; m[6] = 0; m[10] = 1;
        mtx[0] * (mtx[5] * mtx[10] - mtx[6] * mtx[9])
        - mtx[4] * (mtx[1] * mtx[10] - mtx[2] * mtx[9])
        + mtx[8] * (mtx[1] * mtx[6] - mtx[2] * mtx[5]));
}

inline void matrixInverse(const Matrix44f &lfs, Matrix44f *rhs)
{
    const float d = matrixDeterminant(lfs);
    *rhs = matrixCofactor(matrixTranspose(lfs)) / d;
}

inline std::ostream& operator<<(std::ostream &out, const Matrix44f &mtx)
{
    out<<mtx.m[0]<<" "<<mtx.m[4]<<" "<<mtx.m[8]<<" "<<mtx.m[12]<<std::endl
       <<mtx.m[1]<<" "<<mtx.m[5]<<" "<<mtx.m[9]<<" "<<mtx.m[13]<<std::endl
       <<mtx.m[2]<<" "<<mtx.m[6]<<" "<<mtx.m[10]<<" "<<mtx.m[14]<<std::endl
       <<mtx.m[3]<<" "<<mtx.m[7]<<" "<<mtx.m[11]<<" "<<mtx.m[15];
    return out;
}

UD_NAMESPACE_END

#endif
