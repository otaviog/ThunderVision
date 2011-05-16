#include "math.hpp"

UD_NAMESPACE_BEGIN

const float Math::PI_D_180 = UD_FPI/180.0f;
const float Math::_180_D_PI = 180.0f/UD_FPI;

int ceilPowerOfTwo(int a)
{
    a = std::abs(a);

    int r = 1;
    while ( r < a )
        r = r << 1;

    return r;
}

bool Math::quadraticEquation(float A, float B, float C, float *r1, float *r2)
{
    const float delta = B*B - 4.0f*A*C;

    if ( delta < 0.0f )
        return false;

    const float R = std::sqrt(delta);
    const float _2A = 2.0f * A;

    *r1 = (-B + R) / _2A;
    *r2 = (-B - R) / _2A;

    return true;
}

bool Math::lowestPositiveQuadraticRoot(float A, float B, float C, float *r)
{
    const float delta = B*B - 4.0f*A*C;
    bool gotRoot;

    if ( delta < 0.0f )
        return false;

    const float R = std::sqrt(delta);
    const float _2A = 2.0f * A;

    float r1 = (-B + R) / _2A;
    float r2 = (-B - R) / _2A;

    if ( r1 > r2 )
        std::swap(r1, r2);

    if ( r1 < 0.0f )
    {
        if ( r2 < 0.0f )
        {
            gotRoot = false;
        }
        else
        {
            gotRoot = true;
            *r = r2;
        }
    }
    else
    {
        gotRoot = true;
        *r = r1;
    }

    return gotRoot;
}

float Math::triangleArea(const Vec3f &v0, const Vec3f &v1, const Vec3f &v2)
{
    const float l[3] = {
        vecLength(v0 - v1),
        vecLength(v1 - v2),
        vecLength(v2 - v0)
    };

    const float s = (l[0] + l[1] + l[2])/2.0f;
    return std::sqrt(s * (s - l[0]) * (s - l[1]) * (s - l[2]));
}

UD_NAMESPACE_END
