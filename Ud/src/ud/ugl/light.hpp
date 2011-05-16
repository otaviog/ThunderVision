#ifndef UD_LIGHT_HPP
#define UD_LIGHT_HPP

#include "../common.hpp"
#include "../math/vec.hpp"
#include "color.hpp"

UD_NAMESPACE_BEGIN

class Matrix44f;

class Light
{
public:
    Color ambient, diffuse, specular, emission;
    Vec4f position;

    Light()
        : ambient(0.5f, 0.5f, 0.5f),
          diffuse(0.5f, 0.5f, 0.5f),
          specular(0.0f, 0.0f, 0.0f),
          position(0.0f, 1.0f, 0.0f, 1.0f) { }

    Light(const Color &ambdiff, const Vec4f &position)
        : ambient(ambdiff), diffuse(ambdiff),
          specular(White), position(position)
    { }
    
    Light(const Color &amb, const Color &diff, const Vec4f &position)
        : ambient(amb), diffuse(diff),
          specular(White), position(position)
    { }
    
    Light(const Color &amb, const Color &diff,
          const Color &specular, const Vec4f &position)
            : ambient(amb), diffuse(diff), specular(specular),
            position(position) { }

    void genViewMatrix(Matrix44f *pViewMtx) const;
    void draw(float radius) const;
    void applyGL(GLenum light) const;

    void translate(float x, float y, float z)
    {
        position[0] += x;
        position[1] += y;
        position[2] += z;
    }
};

UD_NAMESPACE_END

#endif
