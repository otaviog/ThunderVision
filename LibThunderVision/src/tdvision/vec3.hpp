#ifndef UD_VEC3_HPP
#define UD_VEC3_HPP

#include <ostream>
#include <tdvbasic/common.hpp>

TDV_NAMESPACE_BEGIN

/**
 * Simple 3D vector class.
 *
 * @author Otávio Gomes
 */
class Vec3f
{
public:
    /**
     * Garbage constructor.
     */
    Vec3f() { }

    /**
     * Construct with all components with same value.
     * @param l the value
     */
    Vec3f(float l)
    {
        v[0] = v[1] = v[2] = l;
    }

    /**
     * Constructor.
     * @param x_ the x component
     * @param y_ the y component
     * @param z_ the z component
     */
    Vec3f(float x_, float y_, float z_)
    {
        v[0] = x_;
        v[1] = y_;
        v[2] = z_;
    }

    /**
     * Array constructor.
     * @param a the size must be 3
     */
    Vec3f(float a[])
    {
        v[0] = a[0];
        v[1] = a[1];
        v[2] = a[2];
    }

    /**
     * Sets values.
     * @param x_ the x component
     * @param y_ the y component
     * @param z_ the z componen
     */ 
    void set(float _x, float _y, float _z)
    {
        v[0] = _x;
        v[1] = _y;
        v[2] = _z;
    }

    /**
     * Returns a component. Read-only.
     */
    const float& operator[](int i) const
    {
        return v[i];
    }

    /**       
     * Returns a component
     */
    float& operator[](int i)
    {
        return v[i];
    }

    /**
     * Negativates the vector.
     */
    Vec3f operator-() const
    {
        return Vec3f(-v[0], -v[1], -v[2]);
    }

    /**
     * Adds with another vector.
     * @param rhs right side vector
     */
    Vec3f& operator+=(const Vec3f &rhs)
    {
        v[0] += rhs.v[0];
        v[1] += rhs.v[1];
        v[2] += rhs.v[2];
        return *this;
    }
    
    /**
     * Subtracts with another vector.
     * @param rhs right side vector
     */
    Vec3f& operator-=(const Vec3f &rhs)
    {
        v[0] -= rhs.v[0];
        v[1] -= rhs.v[1];
        v[2] -= rhs.v[2];
        return *this;
    }

    /**
     * Multiplies with another vector.
     * @param rhs right side vector
     */    
    Vec3f& operator*=(float rhs)
    {
        v[0] *= rhs;
        v[1] *= rhs;
        v[2] *= rhs;
        return *this;
    }

    /**
     * Divides with another vector.
     * @param rhs right side vector
     */
    Vec3f& operator/=(float rhs)
    {
        v[0] /= rhs;
        v[1] /= rhs;
        v[2] /= rhs;
        return *this;
    }
    
    /**
     * Apply normalization to the vector.
     */ 
    inline void normalize();

    float v[3]; /*< the vector - x, y, z. */
};

/**
 * Adds two vectors.
 */
inline Vec3f operator+(const Vec3f &lfs, const Vec3f &rhs)
{
    return Vec3f(lfs.v[0] + rhs.v[0], lfs.v[1] + rhs.v[1], lfs.v[2] + rhs.v[2]);
}

/**
 * Subtracts two vectors.
 */
inline Vec3f operator-(const Vec3f &lfs, const Vec3f &rhs)
{
    return Vec3f(lfs.v[0] - rhs.v[0], lfs.v[1] - rhs.v[1], lfs.v[2] - rhs.v[2]);
}

/**
 * Divides a vector by a scalar.
 */
inline Vec3f operator/(const Vec3f &lfs, float rhs)
{
    return Vec3f(lfs.v[0] / rhs, lfs.v[1] / rhs, lfs.v[2] / rhs);
}

/**
 * Divides a vector by a scalar.
 */
inline Vec3f operator/(float rhs, const Vec3f &lfs)
{
    return Vec3f(rhs / lfs.v[0], rhs / lfs.v[1], rhs / lfs.v[2]);
}

/**
 * Multiplies a vector by a scalar.
 */
inline Vec3f operator*(const Vec3f &lfs, float rhs)
{
    return Vec3f(lfs.v[0] * rhs, lfs.v[1] * rhs, lfs.v[2] * rhs);
}

/**
 * Multiplies a vector by a scalar.
 */
inline Vec3f operator*(float rhs, const Vec3f &lfs)
{
    return lfs * rhs;
}

/**
 * Vector scalar product.
 */
inline float vecDot(const Vec3f &lfs, const Vec3f &rhs)
{
    return lfs.v[0] * rhs.v[0] + lfs.v[1] * rhs.v[1] + lfs.v[2] * rhs.v[2];
}

/**
 * Vector cross product.
 */
inline Vec3f vecCross(const Vec3f &lfs, const Vec3f &rhs)
{
    return Vec3f(lfs.v[1] * rhs.v[2] - lfs.v[2] * rhs.v[1],
                        lfs.v[2] * rhs.v[0] - lfs.v[0] * rhs.v[2],
                        lfs.v[0] * rhs.v[1] - lfs.v[1] * rhs.v[0]);
}

/**
 * Vector length.
 */
inline float vecLength(const Vec3f &lfs)
{
    return std::sqrt(lfs[0] * lfs[0] + lfs[1] * lfs[1] + lfs[2] * lfs[2]);
}

/**
 * Vector normal.
 */
inline Vec3f vecNormal(const Vec3f &lfs)
{
    const float len = vecLength(lfs);
    return Vec3f(lfs.v[0]/len, lfs.v[1]/len, lfs.v[2]/len);
}

void Vec3f::normalize()
{
    *this = vecNormal(*this);
}

TDV_NAMESPACE_END

#endif
