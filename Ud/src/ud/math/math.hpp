#ifndef UD_MATH_HPP
#define UD_MATH_HPP

#include <cmath>
#include "vec.hpp"
#include "vec.hpp"
#include "vec.hpp"
#include "quat.hpp"
#include "matrix22.hpp"
#include "matrix33.hpp"
#include "matrix44.hpp"

UD_NAMESPACE_BEGIN

#ifdef __SUNPRO_CC
inline bool hgIsNan(float t)
{
    return t != t;
}
#elif _MSC_VER
#include <float.h>
inline bool hgIsNan(float t)
{
    return _isnan(t) != 0;
}
#else
inline bool hgIsNan(float t)
{
    return std::isnan(t);
}
#endif

#define UD_EPSILON 0.00001
#define UD_FLOAT_EPSILON 0.00001f
#define UD_DOUBLE_EPSILON 0.00001

int ceilPowerOfTwo(int a);

inline bool fltEq(float lfs, float rhs, const float ep = UD_FLOAT_EPSILON)
{
    return std::abs(lfs - rhs) < ep;
}

inline bool fltGt(float lfs, float rhs, const float ep = UD_FLOAT_EPSILON)
{
    return lfs > rhs + ep;
}

inline bool fltGe(float lfs, float rhs, const float ep = UD_FLOAT_EPSILON)
{
    return fltGt(lfs, rhs, ep) || fltEq(lfs, rhs, ep);
}

inline bool fltLt(float lfs, float rhs, const float ep = UD_FLOAT_EPSILON)
{
    return lfs < rhs - ep;
}

inline bool fltLe(float lfs, float rhs, const float ep = UD_FLOAT_EPSILON)
{
    return fltLt(lfs, rhs, ep) || fltEq(lfs, rhs, ep);
}

class Math
{
public:
    /**
     * Solves a quadratic equation of the form A*x^2 + B*x + C.
     * If B*B - 4*A*C is negative, then it returns false, else
     * it's return true and the 2 roots by the pointers r1 and r2.
     *
     * @param A variable that multiplies x^2
     * @param B variable that multiplies x
     * @param C indepedent term
     * @param r1 returns the first root
     * @param r2 returns the second root
     * @return true if got a non negative delta.
     */
    static bool quadraticEquation(float A, float B, float C, float *r1, float *r2);

    static bool lowestPositiveQuadraticRoot(float A, float B, float C, float *r);

    /**
     * Converts degrees to radians.
     * @param deg the angle in degrees
     * @return the angle in radians
     */
    static inline float degToRad(float deg)
    {
        return deg * PI_D_180;
    }

    /**
     * Converts radians to degrees.
     * @param rad the angle in radians
     * @return the angle in degrees
     */
    static inline float radToDeg(float rad)
    {
        return rad * _180_D_PI;
    }

    /**
     * Return the squared value, v*v, just for convinience.
     * @param v the value
     * @return v*v
     */
    static inline float square(const float v)
    {
        return v * v;
    }

    static inline float clamp(float v, float min, float max)
    {
        if ( v < min )
            return min;

        if ( v > max )
            return max;

        return v;
    }
    
    static inline bool inRegion(float v, float min_, float max_)
    {
        return (v >= min_) && (v <= max_);
    }

    static float triangleArea(const Vec3f &v0, const Vec3f &v1, const Vec3f &v2);

#if 0
    /**
     * Fast inverse square root function, was get from:
     * http://www.beyond3d.com/content/articles/8/
     */
    static inline float fastInvSqrt(float x)
    {
        float xhalf = 0.5f*x;
        int i = *(int*)&x;
        i = 0x5f3759df - (i>>1);
        x = *(float*)&i;
        x = x*(1.5f - xhalf*x*x);
        return x;
    }
#endif

    static inline float round(float v)
    {
        return std::floor(v+0.5f);
    }
    static const float PI_D_180;
    static const float _180_D_PI;
};

UD_NAMESPACE_END

#endif
