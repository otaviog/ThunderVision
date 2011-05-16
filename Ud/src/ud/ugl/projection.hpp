#ifndef UD_PROJECTION_HPP
#define UD_PROJECTION_HPP

#include "../common.hpp"

UD_NAMESPACE_BEGIN

class Aabb;
class Vec3f;
class Matrix44f;

class Projection
{
public:
    Projection() { }
    Projection(float _left, float _right, float _bot, float _top,
                    float _near, float _far);

    Projection(float fov, float aspect, float near, float far);

    /**
     * Extract the four conners of the near and far plane.
     * The conners are returned in anti-clock wise:
     * Left Top, Left Bottom, Right Bottom and Right Bottom.
     *
     * @param nearConners the returned near conners
     * @param farConners the returned far conners
     */
    void extractCorners(Vec3f nearConners[4], Vec3f farConners[4]) const;

    void extractAabb(Aabb *box) const;

    void genMatrix(Matrix44f *pMtx) const;

    /**
     * Split the projection in n parts along the z axis.
     * @param parts the split parts in range [0.0, 1.0].
     * Ex. 0.5 specify the middle of projection
     * @param nparts the size of previous array
     * @param out the splited projections
     */
    void split(const float parts[], size_t nparts, Projection out[], float epsilon=0.0f) const;

    /**
     * Applies the projection to the current OpenGL matrix.
     */
    void applyGL() const;

    float getFov() const
    { return m_fov; }

    float getAspect() const
    { return m_aspect; }

    float farD() const
    { return m_farD; }

    float nearD() const
    { return m_nearD; }

    float left() const
    { return m_left; }

    float right() const
    { return m_right; }

    float bottom() const
    { return m_bottom; }

    float top() const
    { return m_top; }

    float farLeft() const
    { return m_farLeft; }

    float farRight() const
    { return m_farRight; }

    float farBottom() const
    { return m_farBottom; }

    float farTop() const
    { return m_farTop; }

private:
    float m_farD, m_nearD, m_left, m_right, m_bottom, m_top,
        m_farLeft, m_farRight, m_farTop, m_farBottom;

    float m_fov, m_aspect;
};

UD_NAMESPACE_END

#endif // UD_PROJECTIONPARMS_HPP
