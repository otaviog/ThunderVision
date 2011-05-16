#ifndef UD_BONDINGSPHERE_HPP
#define UD_BONDINGSPHERE_HPP

#include <cmath>
#include "../common.hpp"
#include "../math/plane.hpp"
#include "../math/vec.hpp"
#include "../math/matrix44.hpp"

UD_NAMESPACE_BEGIN

/**
 * Bounding sphere.
 * @author Otávio Gomes
 */ 
class BoundingSphere
{
public:
    /**
     * Creates a sphere in the orign and with radius = -1.
     */
    BoundingSphere()
            : m_center(0.0f)
    {
        m_radius = -1.0f;
    }

    /**
     * Constructor.
     * @param center center of the sphere
     * @param radius radius
     */
    BoundingSphere(const Vec3f &center, float radius)
            : m_center(center)
    {
        m_radius = radius;
    }

    /**
     * Constructor with radius = -1.
     * @param center center of the sphere
     */
    explicit BoundingSphere(const Vec3f &center)
            : m_center(center)
    {
        m_radius = -1.0f;
    }

    /**
     * Adds a point for inside the sphere.
     * @param v point
     */
    void add(const Vec3f &v)
    {
        const float vMinusCenterLen = vecLength(v - m_center);

        if ( vMinusCenterLen > m_radius )
            m_radius = vMinusCenterLen;
    }
    
    /**
     * Adds a sphere.    
     */
    void add(const BoundingSphere &sphere)
    {
        const float sCenterMinusCenterLen = vecLength(sphere.center() - m_center);
        if ( sCenterMinusCenterLen > m_radius )
        {
            m_center = (sphere.center() + m_center)/2.0f;
            m_radius = sCenterMinusCenterLen + hgMax(sphere.radius(), m_radius);
        }
        else if ( m_radius < sphere.radius() )
        {
            *this = sphere;
        }
    }
    
    /**
     * Sets the center.
     */
    void center(const Vec3f &center)
    {
        m_center = center;
    }

    /**
     * Returns the center.
     */
    const Vec3f& center() const
    {
        return m_center;
    }

    /**
     * Returns the radius.
     */
    float radius() const
    {
        return m_radius;
    }

    /**
     * Sets the radius for -1. 
     */
    void clearRadius()
    {
        m_radius = -1.0f;
    }

    /**
     * Translate the sphere center.
     */
    void translate(const Vec3f &trans)
    {
        m_center += trans;
    }

    /**
     * Transform the sphere center by a matrix.
     * Note: we got no support for scaling bounding boxes.
     */
    void transformCenter(const Matrix44f &mtx)
    {
        m_center = m_center * mtx;        
    }
        
    /**
     * Determines in which side of the plane this sphere is located.    
     */
    PlaneSide classifyPlaneSphere(const Planef &plane) const
    {
        const float dist = plane.distance(m_center);

        if ( std::fabs(dist) < m_radius )
            return Spanning;

        if ( dist > 0.0f )
            return Front;

        if ( dist < 0.0f )
            return Back;

        return Coplanar;
    }

private:
    Vec3f m_center;
    float m_radius;
};

UD_NAMESPACE_END

#endif
