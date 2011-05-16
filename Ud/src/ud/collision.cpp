#include "collision.hpp"

UD_NAMESPACE_BEGIN

namespace collision
{
    /*
     * Static Methods
     */
    const float NoIntersection = std::numeric_limits<float>::infinity();

    /**
     * Mathematics for 3D Game Programming & Computer Graphics
     */
    bool pointInTriangle(
        const Vec3f &point, const Vec3f &trigNormal,
        const Vec3f &p0, const Vec3f &p1,
        const Vec3f &p2)
    {
        const Vec3f Q(p1 - p0);
        const Vec3f R(p2 - p0);
    
        const float QQ = vecDot(Q, Q);
        const float QR = vecDot(Q, R);
        const float RR = vecDot(R, R);

        const float _1_det = 1.0f/(QQ*RR - QR*QR);

        const float iQQ = QQ*_1_det;
        const float iQR = QR*_1_det;
        const float iRR = RR*_1_det;
    
        const Vec3f Z(point - p0);
    
        const float ZQ = vecDot(Z, Q);
        const float ZR = vecDot(Z, R);
    
        const float w1 = iQQ*ZQ + iQR*ZR;
        const float w2 = iQR*ZQ + iRR*ZR;
        const float w3 = 1.0f - w1 - w2;

        if ( Math::inRegion(w1, 0.0f, 1.0f) && Math::inRegion(w2, 0.0f, 1.0f)
             && Math::inRegion(w3, 0.0f, 1.0f) )    
            return true;
        else
            return false;
    }

    bool sphereSphereCollision(const BoundingSphere &M,
                               const Vec3f &vel,
                               const BoundingSphere &S)
    {
        const Vec3f e = M.center() - S.center();
        const float r = S.radius() + M.radius();

        if ( vecLength(e) < r )
            return true;

        const float delta = Math::square(vecDot(e, vel))
            - vecDot(vel, vel) * (vecDot(e, e) - r*r);
        if ( delta < 0.0f )
            return false;

        const float t = (-vecDot(e, vel) - std::sqrt(delta)) / vecDot(vel, vel);
        if ( t < 0.0f || t > 1.0f )
            return false;

        return true;
    }

    float rayTriangleCollision(
        const Vec3f &rayOrign, const Vec3f &rayDir,
        const Vec3f &p0, const Vec3f &p1,
        const Vec3f &p2)
    {
        Planef trigPlane(p0, p1, p2);
        float div = vecDot(rayDir, trigPlane.normal);

        if ( div == 0.0f )
            return NoIntersection;

        const float t = -(vecDot(rayOrign, trigPlane.normal) + trigPlane.d) / div;

        if ( t < 0.0f )
            return NoIntersection;

        Vec3f interscPoint = rayOrign  + rayDir * t;

        if ( pointInTriangle(interscPoint, rayOrign, p0, p1, p2) == true )
            return t;
        else
            return NoIntersection;
    }

    float sphereTriangleCollision(
        const BoundingSphere &sphere,
        const Vec3f &dir,
        const Vec3f &p0, const Vec3f &p1,
        const Vec3f &p2, Vec3f *contactPoint)
    {
        Planef trigPlane(p0, p1, p2);
        float d = vecDot(dir, trigPlane.normal);
        float minDist = NoIntersection;

        if  ( vecDot(dir, trigPlane.normal) > 0.0 )
        {
            return NoIntersection;
        }
    
        if ( d == 0.0f )
        {
            if ( trigPlane.distance(sphere.center()) < sphere.radius() )
                return NoIntersection;
        }
        else 
        {
            const Vec3f orign = sphere.center()
                - sphere.radius()*vecNormal(trigPlane.normal);
            const float t = -(trigPlane.d + vecDot(orign, trigPlane.normal)) / d;

            if ( t >= 0.0f )
            {
            
                const Vec3f planePoint = orign + dir * t;
                if ( pointInTriangle(planePoint,
                                     vecNormal(trigPlane.normal),
                                     p0, p1, p2) )
                {
                    *contactPoint = planePoint;
                    return t;
                }
            }
        }

        float dist = spherePointDistance(sphere, dir, p0);
        if ( dist < minDist )
        {
            minDist = dist;
            *contactPoint = p0;
        }

        dist = spherePointDistance(sphere, dir, p1);
        if ( dist < minDist )
        {
            minDist = dist;
            *contactPoint = p1;
        }

        dist = spherePointDistance(sphere, dir, p2);
        if ( dist < minDist )
        {
            minDist = dist;
            *contactPoint = p2;
        }

        Vec3f edgeContactPoint;
        dist = sphereEdgeDistance(sphere, dir, p1, p0, &edgeContactPoint);
        if ( dist < minDist )
        {
            minDist = dist;
            *contactPoint = edgeContactPoint;
        }

        dist = sphereEdgeDistance(sphere, dir, p2, p1, &edgeContactPoint);
        if ( dist < minDist )
        {
            minDist = dist;
            *contactPoint = edgeContactPoint;
        }

        dist = sphereEdgeDistance(sphere, dir, p0, p2, &edgeContactPoint);
        if ( dist < minDist )
        {
            minDist = dist;
            *contactPoint = edgeContactPoint;
        }

        return minDist;
    }

    float spherePointDistance(const BoundingSphere &sphere,
                              const Vec3f &D,
                              const Vec3f &P)
    {
        float distance;

        const Vec3f E = sphere.center() - P;
        if ( !Math::lowestPositiveQuadraticRoot(
                 vecDot(D, D),
                 2.0f*vecDot(D, E),
                 vecDot(E, E) - sphere.radius()*sphere.radius(),
                 &distance) )
        {
            return NoIntersection;
        }

        return distance;
    }

    float sphereEdgeDistance(
        const BoundingSphere &sphere,
        const Vec3f &D,
        const Vec3f &A,
        const Vec3f &B,
        Vec3f *contactPoint)
    {
        const Vec3f AB = B - A;
        const Vec3f C = sphere.center() - A;
        const float AB2 = vecDot(AB, AB);
        const float AB_dot_C = vecDot(C, AB);
        const float AB_dot_D = vecDot(AB, D);

        float minR;
        if ( !Math::lowestPositiveQuadraticRoot(
                 vecDot(D, D) - Math::square(AB_dot_D)/AB2,
                 2.0f*(vecDot(C, D) - (AB_dot_D*AB_dot_C)/AB2),
                 vecDot(C, C) - Math::square(sphere.radius())
                 - Math::square(AB_dot_C)/AB2,
                 &minR) )
        {
            return NoIntersection;
        }

        const ud::Vec3f intersect = sphere.center() + D*minR - A; 
        const float t = vecDot(AB, intersect);
        if ( 0.0f <= t && t <= AB2 )
        {
            *contactPoint = A + AB * t;
            return minR;
        }
        else
        {
            return NoIntersection;
        }

    }
}

UD_NAMESPACE_END
