#ifndef UD_PLANE_HPP
#define UD_PLANE_HPP

#include <ostream>
#include "../common.hpp"
#include "vec.hpp"

UD_NAMESPACE_BEGIN

enum PlaneSide
{
    Front,
    Back,
    Coplanar,
    Spanning
};

/**
 * Defines a A*x + B*y + C*z + D plane. It's stores the normal (x, y, z) and D.
 */
class Planef
{
public:
    /**
     * Empty constructor, initialized with unknown values.
     */
    Planef() { }

    /**
     * Direct initialize.
     * @param x normal x
     * @param y normal y
     * @param z normal z
     * @param D plane distance
     */
    Planef(float x, float y, float z, float D)
            : normal(x, y, z)
    {
        d = D;
    }

    /**
     * Direct initialize.
     * @param normal plane normal
     * @param D plane distance
     */
    Planef(const Vec3f &normal, float D)
            : normal(normal)
    {
        d = D;
    }

    /**
     * Direct initialize the normal. Calculate D using a point in plane.
     * @param normal plane normal
     * @param p a point in the plane
     */
    Planef(const Vec3f &normal, const Vec3f &p)
            : normal(normal)
    {
        d = -vecDot(normal, p);
    }

    /**
     * Creates a plane from a triangle, the plane normal is not normalized.
     * Assumes that the points are in counter clock wise order (OpenGL).
     * @param a triangle point
     * @param b triangle point
     * @param c triangle point
     */
    Planef(const Vec3f &a, const Vec3f &b, const Vec3f &c)
    {
        normal = vecCross(b - a, c - b);
        d = -vecDot(normal, a);
    }

    /**
     * Classify a point in respect with the plane.
     * @param point any point
     * @param ep epsilon
     */
    PlaneSide classifyPoint(const Vec3f &point, const float ep = 0) const
    {
        const float dist(vecDot(normal, point) + d);
        PlaneSide ret;

        if ( dist > ep )
            ret = Front;
        else if ( dist < -ep )
            ret = Back;
        else
            ret = Coplanar;

        return ret;
    }

    /**
     * Classify a point in respect with the plane. Homogeneous coordinates version.
     * @param point any point
     * @param ep epsilon
     */
    PlaneSide classifyPoint(const Vec4f &point, const float ep = 0) const
    {
        const float d(normal[0] * point[0]
                     + normal[1] * point[1]
                     + normal[2] * point[2]
                     + d * point[3]);
        PlaneSide ret;

        if ( d > ep )
            ret = Front;
        else if ( d < -ep )
            ret = Back;
        else
            ret = Coplanar;

        return ret;
    }

    PlaneSide classifyTriangle(const Vec3f &p0, const Vec3f &p1, 
                               const Vec3f &p2, const float ep = 0) const
    {
        const PlaneSide r1 = classifyPoint(p0, ep);
        const PlaneSide r2 = classifyPoint(p1, ep);
        const PlaneSide r3 = classifyPoint(p2, ep);

        if ( r1 == Coplanar && r2 == Coplanar && r3 == Coplanar )
            return Coplanar;

        if ( r1 != r2 || r1 != r3 )
            return Spanning;
    
        return r1;
    }
    
    /**
     * Returns the distance from a point to the plane.
     * @param p any point
     */
    float distance(const Vec3f &p) const
    {
        return vecDot(normal, p) + d;
    }

    /**
     * Given the line begin p0 and the line end p1:
     * p0 + (p1 - p0)*t <br>
     * Returns the t where the above equation intersects the plane.
     * Is assumed that the line intersects the plane.
     * @param p0 the line start point
     * @param p1 the line end point
     */
    float linearInterp(const Vec3f &p0, const Vec3f &p1) const
    {
        const Vec3f v(p1-p0);
        return (-vecDot(p0, normal) - d)/vecDot(v, normal);
    }

    /**
     * Given:
     * orign + dir*t
     * Returns the t where the above equation intersects the plane.
     * Is assumed that the ray intersects the plane.
     * @param orign the ray origin
     * @param dir the ray direction
     */
    float intersectionRayT(const Vec3f &orign, const Vec3f &dir) const
    {
        return (-vecDot(orign, normal) - d)/vecDot(dir, normal);
    }

    /**
     * Returns the point of intersection between a ray and the plane.
     * @param orign the ray origin
     * @param dir the ray direction
     */
    Vec3f intersectionRay(const Vec3f &orign, const Vec3f &dir) const
    {
        return orign + dir*intersectionRayT(orign, dir);
    }

    /**
     * Returns the point of intersection between a line and the plane.
     * @param p0 the line start point
     * @param p1 the line end point
     */
    Vec3f intersectionPoint(const Vec3f &p0, const Vec3f &p1) const
    {
        const Vec3f v(p1-p0);
        return intersectionRay(p0, v);
    }

    /**
     * Returns the point of intersection between a line and the plane. Homogeneous coordinates version.
     * @param p0 the line start point
     * @param p1 the line end point
     */
    Vec4f intersectionPoint(const Vec4f &p0, const Vec4f &p1) const
    {
        const Vec4f v(p1-p0);

        const float p0_dot_normal(-(p0[0] * normal[0]
                                     + p0[1] * normal[1]
                                     + p0[2] * normal[2])
                                   - p0[3] * d);
        float v_dot_normal(v[0] * normal[0]
                          + v[1] * normal[1]
                          + v[2] * normal[2]
                          + v[3] * d);

        const float t(p0_dot_normal / v_dot_normal);

        return p0 + v*t;
    }

    Vec3f intersectionPoint(const Planef &p1, const Planef &p2)
    {
        const Vec3f p = d * vecCross(p1.normal, p2.normal)
            + p1.d * vecCross(p2.normal, normal)
            + p2.d * vecCross(normal, p1.normal);

        const float d = -1/vecDot(normal, vecCross(p1.normal, p2.normal));

        return p * d;
    }

    Vec3f normal;
    float d;
};

inline std::ostream& operator<<(std::ostream& out, const Planef &plane)
{
    out<<plane.d<<" "<<plane.normal;
    return out;
}

UD_NAMESPACE_END

#endif
