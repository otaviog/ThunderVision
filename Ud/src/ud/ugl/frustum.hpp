#ifndef UD_FRUSTUM_HPP
#define UD_FRUSTUM_HPP

#include <ostream>
#include "../common.hpp"
#include "../math/plane.hpp"
#include "../math/vec.hpp"
#include "../math/matrix44.hpp"

UD_NAMESPACE_BEGIN

class Projection;
class Aabb;
class BoundingSphere;
class Camera;

/**
 * Represents the view pyramid. 
 */
class Frustum
{
public:
    /**
     * Frustum plane enumeration.
     */
    enum CullPlane
    {
        Left = 0,
        Right = 1,
        Top = 2,
        Bottom = 3,
        Near = 4,
        Far = 5
    };

    /**
     * Frustum conner enumeration. Means [Left|Right][Top|Bottom][Near|Front].
     */
    enum Conners
    {
        LBN = 0, LTN = 1, LBF = 2, LTF = 3,
        RBN = 4, RTN = 5, RBF = 6, RTF = 7
    };

    /**
     * Garbage constructor or extracts the current OGL state frustum, depending on the extract parameter.
     * @param extrat if <code>false</code> nothing is done, leaving the values with garbage
     *               if <code>true</code> extracts the current OGL state frustum using the modelview and projection
     */
    Frustum(bool extract=false)
    {
        if ( extract )
            extractFromOGLState();
    }

    /**
     * Construct the frustum from a given projection*modelview matrix.
     * @param viewProjectionMatrix modelview projection. Note project must be multiplied first 
     */
    Frustum(const Matrix44f &viewProjectionMatrix)
    {
        extract(viewProjectionMatrix);
    }
    
    Frustum(const Matrix44f &proj, const Matrix44f &view)
    {
        extract(proj*view);
    }

    Frustum(const Matrix44f &view, const Projection &proj)
    {
        extract(view, proj);
    }

    Frustum(const Camera &camera, const Projection &proj);

    void extract(const Matrix44f &viewProjectionMatrix);
    void extract(const Matrix44f &viewMatrix, const Projection &proj);
    void extractFromOGLState();

    void draw() const;

    bool isInside(float x, float y, float z) const
    {
        return isInside(Vec3f(x, y, z));
    }

    bool isInside(const Vec3f &p) const;
    bool isInside(const Aabb &box) const;
    bool isInside(const BoundingSphere &sphere) const;

    Planef& getPlane(CullPlane cp)
    {
        return m_planes[cp];
    }

    const Planef& getPlane(CullPlane cp) const
    {
        return m_planes[cp];
    }

    const Vec3f* getConners() const
    { return m_conners; }

    void getConners(Vec3f cs[8]) const;
    void extractAabb(Aabb *box) const;

    static void split(const Matrix44f &viewMatrix,
                      const Projection &proj,
                      const float parts[], size_t nparts,
                      Projection projout[],
                      Frustum frusout[], float epsilon=0.0f);
private:
    Planef m_planes[6];
    Vec3f m_conners[8];
};

std::ostream& operator<<(std::ostream &out, const Frustum &frus);

UD_NAMESPACE_END

#endif
