#ifndef UD_AABB_HPP
#define UD_AABB_HPP

// Basead on 3D Math Primer for Graphics and Game Development
#include <limits>
#include "../common.hpp"
#include "../math/matrix44.hpp"
#include "../math/vec.hpp"

UD_NAMESPACE_BEGIN

/**
 * Axis aligned bounding box class.
 *
 * @author Otávio Gomes
 */
class Aabb
{
public:
    /**
     * Creates a box with infinite inverse boundings (can accept any point).
     */
    Aabb()
    {
        clear();
    }

    /**
     * Constructor.
     * @param bmin minimum point
     * @param bmax maximum point
     */
    Aabb(const Vec3f &bmin, const Vec3f &bmax)
        : m_min(bmin), m_max(bmax) { }

    /**
     * Returns the minumum point.
     */
    const Vec3f& getMin() const
    {
        return m_min;
    }

    /**
     * Returns the maximum point.
     */
    const Vec3f& getMax() const
    {
        return m_max;
    }

    /**
     * Sets the minimum point.
     */
    void setMin(const Vec3f &v)
    {
        m_min = v;
    }

    /**
     * Sets the maximum point.
     */
    void setMax(const Vec3f &v)
    {
        m_max = v;
    }

    /**
     * Computes the box half vectors.
     */
    void halfVectors(Vec3f *hx, Vec3f *hy, Vec3f *hz) const
    {
        const Vec3f d = (m_max - m_min)/2.0f;

        *hx = Vec3f(d[0], 0.0f, 0.0f);
        *hy = Vec3f(0.0f, d[1], 0.0f);
        *hz = Vec3f(0.0f, 0.0f, d[2]);
    }

    /**
     * Computes the center of the box.
     */
    Vec3f center() const
    {
        return (m_min + m_max) * 0.5f;
    }

    /**
     * Test if a point indice of the box.
     */
    bool isInside(const Vec3f &p) const
    {
        return (p[0] >= m_min[0] && p[0] <= m_max[0])
            && (p[1] >= m_min[1] && p[1] <= m_max[1])
            && (p[2] >= m_min[2] && p[2] <= m_max[2]);
    }

    /**
     * Adds a point to the box. If it's out of the boundaries, the box will grow.
     */
    void add(const Vec3f &p)
    {
        if ( p[0] < m_min[0] )
            m_min[0] = p[0];
        if ( p[1] < m_min[1] )
            m_min[1] = p[1];
        if ( p[2] < m_min[2] )
            m_min[2] = p[2];

        if ( p[0] > m_max[0] )
            m_max[0] = p[0];
        if ( p[1] > m_max[1] )
            m_max[1] = p[1];
        if ( p[2] > m_max[2] )
            m_max[2] = p[2];
    }

    /**
     * Adds a point to the box. If it's out of the boundaries, the box will grow.
     */
    void add(float x, float y, float z)
    {
        add(Vec3f(x, y, z));
    }

    /**
     * Adds a box to the box. If it's out of the boundaries, the box will grow.
     */
    void add(const Aabb &box)
    {
        add(box.getMin());
        add(box.getMax());
    }

    /**
     * Transforms by a matrix.
     * Created by 3D Math Primer for Graphics and Game Development.
     */
    void transform(const Matrix44f &mtx);

    /**
     * Translate the box.
     */
    void translate(const Vec3f &trans)
    {
        m_max += trans;
        m_min += trans;
    }

    /**
     * Sets the box center.
     */
    void center(const Vec3f &newCenter)
    {
        m_max = (m_max + newCenter) * 0.5f;
        m_min = (m_min + newCenter) * 0.5f;
    }

    /**
     * Extracts all box points.
     * @param points returned array.
     */
    void extractPoints(Vec3f points[8]) const;

    /**
     *
     */
    Vec3f maxLookUp(const Vec3f &n) const
    {
        Vec3f r;

        for (int k=0; k<3; k++)
        {
            if ( n[k] > 0 )
                r[k] = m_max[k];
            else
                r[k] = m_min[k];
        }

        return r;
    }

    /**
     *
     */
    Vec3f minLookUp(const Vec3f &n) const
    {
        Vec3f r;

        for (int k=0; k<3; k++)
        {
            if ( n[k] > 0 )
                r[k] = m_min[k];
            else
                r[k] = m_max[k];
        }

        return r;
    }

    /**
     * Sets box with infinite inverse boundings (can accept any point).
     */
    void clear()
    {
        m_min = Vec3f(std::numeric_limits<float>::infinity());
        m_max = Vec3f(-std::numeric_limits<float>::infinity());
    }

    /**
     * Draws the box for debugging matter.
     */
    void draw() const;

    /**
     * Split the box within the z axis.
     * Experimental.
     */
    Aabb splitZ(float startU, float endU) const;

private:
    Vec3f m_min, m_max;
};

UD_NAMESPACE_END

#endif
