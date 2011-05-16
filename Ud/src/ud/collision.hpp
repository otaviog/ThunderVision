#ifndef UD_COLLISION_HPP
#define UD_COLLISION_HPP

#include "../common.hpp"

UD_NAMESPACE_BEGIN

namespace collision
{
    static const float NoIntersection;
    
    /**
     * Tests if a point is inside a triangle.
     * @param point point to test
     * @param trigNormal triangle's normal
     * @param p0 triangle first point
     * @param p1 triangle second point
     * @param p2 triangle thrird point
     */
    static bool pointInTriangle(const Vec3f &point,
                                const Vec3f &trigNormal,
                                const Vec3f &p0,
                                const Vec3f &p1,
                                const Vec3f &p2);

    /**
     * Test if a ray collides with a triangles.
     * Returns NoIntersection if the ray never pass by the triangle plane,
     * returns > 1.0 if the ray not collides but it can.
     * returns < 1.0 if the ray will collide.
     * @param rayOrign the ray orign point
     * @param rayDirection the ray velocity
     * @param p0 triangle first point
     * @param p1 triangle second point
     * @param p2 triangle thrird point
     */
    static float rayTriangleCollision(
        const Vec3f &rayOrign,
        const Vec3f &rayDirection,
        const Vec3f &p0,
        const Vec3f &p1,
        const Vec3f &p2);

    /**
     * Returns NoIntersection if the sphere never pass by the point,
     * returns > 1.0 if the sphere not collides but it can.
     * returns < 1.0 if the sphere will collide.
     * @param sphere the sphere
     * @param velocity sphere's velocity
     * @param P the point
     */
    static float spherePointDistance(
        const BoundingSphere &sphere,
        const Vec3f &velocity,
        const Vec3f &P);
    
    /**
     * Returns NoIntersection if the sphere never pass by the edge,
     * returns > 1.0 if the sphere not collides but it can.
     * returns < 1.0 if the sphere will collide.
     * @param sphere the sphere
     * @param velocity sphere's velocity
     * @param A the edge's first point
     * @param B the edge's second point
     * @param contactPoint returns where the sphere collided.
     */
    static float sphereEdgeDistance(
        const BoundingSphere &sphere,
        const Vec3f &velocity,
        const Vec3f &A,
        const Vec3f &B,
        Vec3f *contactPoint);

    /**
     * Returns if a moving sphere will collide with a static sphere.
     */
    static bool sphereSphereCollision(
        const BoundingSphere &movSphere,
        const Vec3f &velocity,
        const BoundingSphere &stacSphere);

    /**
     * Test for sphere-triangle collision, returns the distance between the contact
     * point to the sphere or instead returns Collision::NoIntersection if they never enter in contact.
     *
     * @param sphere the bounding sphere
     * @param velocity the vector representing the direction and velocity
     * @param p0 an vertex of the triangle
     * @param p1 an vertex of the triangle
     * @param p2 an vertex of the triangle
     * @param contactPoint if the sphere collides with the triangle then this function
     *           returns the contact point
     * @return the distance between the contact point and the sphere
     */
    static float sphereTriangleCollision(
        const BoundingSphere &sphere,
        const Vec3f &velocity,
        const Vec3f &p0, const Vec3f &p1,
        const Vec3f &p2, Vec3f *contactPoint);

}

UD_NAMESPACE_END

#endif /* UD_COLLISION_HPP */
