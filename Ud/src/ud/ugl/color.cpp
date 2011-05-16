#include "color.hpp"

UD_NAMESPACE_BEGIN

const Color White = Color(1.0f, 1.0f, 1.0f, 1.0f);
const Color Black = Color(0.0f, 0.0f, 0.0f, 0.0f);
const Color Red = Color(1.0f, 0.0f, 0.0f, 1.0f);
const Color Blue = Color(0.0f, 0.0f, 1.0f, 1.0f);
const Color Green = Color(0.0f, 1.0f, 0.0f, 1.0f);
const Color Grey = Color(0.5f, 0.5f, 0.5f, 1.0f);
const Color Yellow = Color(1.0f, 1.0f, 0.0f, 1.0f);

Color::Color(float red, float green, float blue, float alpha)
{
    R(red);
    G(green);
    B(blue);
    A(alpha);
}

Color::Color(Uint8 red, Uint8 green,
             Uint8 blue, Uint8 alpha)
{
    R(red);
    G(green);
    B(blue);
    A(alpha);
}

Color::Color(unsigned long color)
{
    const Uint8 alpha((color & 0xFF000000) >> 24);
    const Uint8 red  ((color & 0x00FF0000) >> 16);
    const Uint8 green((color & 0x0000FF00) >> 8);
    const Uint8 blue (color & 0x000000FF);

    R(red);
    G(green);
    B(blue);
    A(alpha);
}

float Color::ucharToFloat(unsigned char value)
{
    float result = (float) value;
    return result/255.0f;
}

Uint8 Color::floatToUChar(float value)
{
    unsigned char result = (unsigned char) value;
    return static_cast<Uint8>(result * 255.0f);
}

UD_NAMESPACE_END
