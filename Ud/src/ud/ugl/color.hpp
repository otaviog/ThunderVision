#ifndef UD_COLOR_HPP
#define UD_COLOR_HPP

#include "../common.hpp"

UD_NAMESPACE_BEGIN

/**
 * Holds RGB color. Arithmetic operators are also provide.
 */
class Color
{
public:
    /**
     * Garbage constructor.
     */
    Color() { }
    
    /**
     * Float constructor.
     * @param red
     * @param green
     * @param blue
     * @param alpha     
     */
    Color(float red, float green, float blue, float alpha=1.0);
    
    /**
     * 8-byte constructor.
     * @param red
     * @param green
     * @param blue
     * @param alpha     
     */
    Color(Uint8 red, Uint8 green,
          Uint8 blue, Uint8 alpha=255);

    /**
     * 4-byte constructor.
     */
    Color(unsigned long color);


    /**
     * Returns a component.
     */
    float operator[](int i) const
    {
        return m_col.a[i];
    }
    
    /**
     * Returns a component.
     */
    float& operator[](int i)
    {
        return m_col.a[i];
    }
    
    void operator*=(const Color &rhs)
    {
        m_col.c.r *= rhs.R();
        m_col.c.g *= rhs.G();
        m_col.c.b *= rhs.B();
        m_col.c.a *= rhs.A();
    }

    void operator+=(const Color &rhs)
    {
        m_col.c.r += rhs.R();
        m_col.c.g += rhs.G();
        m_col.c.b += rhs.B();
        m_col.c.a += rhs.A();

    }

    void operator-=(const Color &rhs)
    {
        m_col.c.r -= rhs.R();
        m_col.c.g -= rhs.G();
        m_col.c.b -= rhs.B();
        m_col.c.a -= rhs.A();
    }

    void operator*=(float t)
    {
        m_col.c.r *= t;
        m_col.c.g *= t;
        m_col.c.b *= t;
        m_col.c.a *= t;
    }

    void operator/=(float t)
    {
        m_col.c.r /= t;
        m_col.c.g /= t;
        m_col.c.b /= t;
        m_col.c.a /= t;
    }

    float* array()
    {
        return m_col.a;
    }

    const float* array() const
    {
        return m_col.a;
    }

    float R() const
    {
        return m_col.c.r;
    }
    float G() const
    {
        return m_col.c.g;
    }
    float B() const
    {
        return m_col.c.b;
    }
    float A() const
    {
        return m_col.c.a;
    }

    Uint8 AC() const
    {
        return floatToUChar(m_col.c.a);
    }
    Uint8 RC() const
    {
        return floatToUChar(m_col.c.r);
    }
    Uint8 GC() const
    {
        return floatToUChar(m_col.c.g);
    }
    Uint8 BC() const
    {
        return floatToUChar(m_col.c.b);
    }

    Uint32 Long() const
    {
        return (AC()<<24)
               | (RC()<<16)
               | (GC()<<8)
               | BC();
    }

    void set(float r, float g, float b)
    {
        R(r);
        G(g);
        B(b);
    }

    void set(float r, float g, float b, float a)
    {
        R(r);
        G(g);
        B(b);
        A(a);
    }

    void R(float red)
    {
        m_col.c.r = red;
    }
    void G(float green)
    {
        m_col.c.g = green;
    }
    void B(float blue)
    {
        m_col.c.b = blue;
    }
    void A(float alpha)
    {
        m_col.c.a = alpha;
    }

    void R(Uint8 red)
    {
        m_col.c.r = ucharToFloat(red);
    }
    void G(Uint8 green)
    {
        m_col.c.g = ucharToFloat(green);
    }
    void B(Uint8 blue)
    {
        m_col.c.b = ucharToFloat(blue);
    }
    void A(Uint8 alpha)
    {
        m_col.c.a = ucharToFloat(alpha);
    }

private:
    static float ucharToFloat(unsigned char value);
    static unsigned char floatToUChar(float value);

    union
    {
        struct
        {
            float r, g, b, a;
        }c;
        float a[4];
    }m_col;

    friend Color operator+(const Color &lfs, const Color &rhs);
    friend Color operator-(const Color &lfs, const Color &rhs);
    friend Color operator*(const Color &lfs, const Color &rhs);
    friend Color operator*(const Color &lfs, float s);
    friend Color operator/(const Color &lfs, float s);
    friend Color colorSaturate(const Color &col);
};

extern const Color White;
extern const Color Black;
extern const Color Red;
extern const Color Blue;
extern const Color Yellow;
extern const Color Green;
extern const Color Grey;

inline Color operator+(const Color &lfs, const Color &rhs)
{
    Color ret;

    ret.m_col.c.r = lfs.m_col.c.r + rhs.m_col.c.r;
    ret.m_col.c.g = lfs.m_col.c.g + rhs.m_col.c.g;
    ret.m_col.c.b = lfs.m_col.c.b + rhs.m_col.c.b;

    return ret;
}

inline Color operator-(const Color &lfs, const Color &rhs)
{
    Color ret;

    ret.m_col.c.r = lfs.m_col.c.r - rhs.m_col.c.r;
    ret.m_col.c.g = lfs.m_col.c.g - rhs.m_col.c.g;
    ret.m_col.c.b = lfs.m_col.c.b - rhs.m_col.c.b;

    return ret;
}

inline Color operator*(const Color &lfs, const Color &rhs)
{
    Color ret;

    ret.m_col.c.r = lfs.m_col.c.r * rhs.m_col.c.r;
    ret.m_col.c.g = lfs.m_col.c.g * rhs.m_col.c.g;
    ret.m_col.c.b = lfs.m_col.c.b * rhs.m_col.c.b;

    return ret;
}

inline Color operator*(const Color &lfs, float s)
{
    Color ret;

    ret.m_col.c.r = lfs.m_col.c.r * s;
    ret.m_col.c.g = lfs.m_col.c.g * s;
    ret.m_col.c.b = lfs.m_col.c.b * s;

    return ret;
}

inline Color operator/(const Color &lfs, float s)
{
    Color ret;

    ret.m_col.c.r = lfs.m_col.c.r / s;
    ret.m_col.c.g = lfs.m_col.c.g / s;
    ret.m_col.c.b = lfs.m_col.c.b / s;

    return ret;
}

inline Color colorSaturate(const Color &col)
{
    Color ret;

    if ( col.m_col.c.r > 1.0f )
        ret.m_col.c.r = 1.0f;
    else
        ret.m_col.c.r = col.m_col.c.r;

    if ( col.m_col.c.g > 1.0f )
        ret.m_col.c.g = 1.0f;
    else
        ret.m_col.c.g = col.m_col.c.g;

    if ( col.m_col.c.b > 1.0f )
        ret.m_col.c.b = 1.0f;
    else
        ret.m_col.c.b = col.m_col.c.b;

    return ret;
}

UD_NAMESPACE_END

#endif
