#include <iostream>
#include "camera.hpp"
#include "aabb.hpp"
#include "boundingsphere.hpp"
#include "projection.hpp"
#include "frustum.hpp"

UD_NAMESPACE_BEGIN

Frustum::Frustum(const Camera &camera, const Projection &proj)
{
    Matrix44f view;
    camera.genMatrix(&view);

    extract(view, proj);
}

void Frustum::extract(const Matrix44f &vp)
{
    m_planes[Left].normal[0] = vp[3] + vp[0];
    m_planes[Left].normal[1] = vp[7] + vp[4];
    m_planes[Left].normal[2] = vp[11] + vp[8];
    m_planes[Left].d = vp[15] + vp[12];

    m_planes[Right].normal[0] = -vp[0] + vp[3];
    m_planes[Right].normal[1] = -vp[4] + vp[7];
    m_planes[Right].normal[2] = -vp[8] + vp[11];
    m_planes[Right].d = -vp[12] + vp[15];

    m_planes[Bottom].normal[0] = vp[3] + vp[1];
    m_planes[Bottom].normal[1] = vp[7] + vp[5];
    m_planes[Bottom].normal[2] = vp[11] + vp[9];
    m_planes[Bottom].d = vp[15] + vp[13];

    m_planes[Top].normal[0] = -vp[1] + vp[3];
    m_planes[Top].normal[1] = -vp[5] + vp[7];
    m_planes[Top].normal[2] = -vp[9] + vp[11];
    m_planes[Top].d = -vp[13] + vp[15];

    m_planes[Far].normal[0] = vp[3] + vp[2];
    m_planes[Far].normal[1] = vp[7] + vp[6];
    m_planes[Far].normal[2] = vp[11] + vp[10];
    m_planes[Far].d = vp[15] + vp[14];

    m_planes[Near].normal[0] = -vp[2] + vp[3];
    m_planes[Near].normal[1] = -vp[6] + vp[7];
    m_planes[Near].normal[2] = -vp[10] + vp[11];
    m_planes[Near].d = -vp[14] + vp[15];

    m_conners[LBN] = m_planes[Left].intersectionPoint(m_planes[Bottom], m_planes[Near]);
    m_conners[LBF] = m_planes[Left].intersectionPoint(m_planes[Bottom], m_planes[Far]);
    m_conners[LTN] = m_planes[Left].intersectionPoint(m_planes[Top], m_planes[Near]);
    m_conners[LTF] = m_planes[Left].intersectionPoint(m_planes[Top], m_planes[Far]);

    m_conners[RBN] = m_planes[Right].intersectionPoint(m_planes[Bottom], m_planes[Near]);
    m_conners[RBF] = m_planes[Right].intersectionPoint(m_planes[Bottom], m_planes[Far]);
    m_conners[RTN] = m_planes[Right].intersectionPoint(m_planes[Top], m_planes[Near]);
    m_conners[RTF] = m_planes[Right].intersectionPoint(m_planes[Top], m_planes[Far]);
}

void Frustum::extract(const Matrix44f &view, const Projection &proj)
{
    const float lnear = -proj.nearD();
    const float lfar = -proj.farD();
    const Vec3f nearNorm(0.0f, 0.0f, -1.0f);
    const Vec3f farNorm(0.0f, 0.0f, 1.0f);

    // Left plane
    const Vec3f ltn(proj.left(), proj.top(), lnear),
        lbn(proj.left(), proj.bottom(), lnear),
        ltf(proj.farLeft(), proj.farTop(), lfar),
        lbf(proj.farLeft(), proj.farBottom(), lfar);

    const Vec3f leftNorm = vecNormal(
        vecCross(lbn - ltn, ltf - ltn));

    // Right plane
    const Vec3f rtn(proj.right(), proj.top(), lnear), // right top near
        rbn(proj.right(), proj.bottom(), lnear), // right bottom near
        rtf(proj.farRight(), proj.farTop(), lfar); // right top far

    const Vec3f rightNorm = vecNormal(vecCross(rtf - rtn, rbn - rtn));

    // Top plane
    const Vec3f topNorm = vecNormal(vecCross(ltf - ltn, rtn - ltn));

    // Bottom plane lbn rbf
    const Vec3f rbf(proj.farRight(), proj.farBottom(), lfar);
    const Vec3f botNorm = vecNormal(vecCross(rbf - rbn, lbn - rbn));

    Matrix44f viewSpace, viewSpaceN;
    matrixInverse(view, &viewSpace);

    viewSpaceN = matrixTranspose(view);

    m_conners[LBN] = lbn * viewSpace;
    m_conners[LTN] = ltn * viewSpace;
    m_conners[LBF] = lbf * viewSpace;
    m_conners[LTF] = ltf * viewSpace;
    m_conners[RBN] = rbn * viewSpace;
    m_conners[RTN] = rtn * viewSpace;
    m_conners[RBF] = rbf * viewSpace;
    m_conners[RTF] = rtf * viewSpace;

    m_planes[Left] = Planef(leftNorm * viewSpaceN, m_conners[LTN]);
    m_planes[Right] = Planef(rightNorm * viewSpaceN, m_conners[RTN]);
    m_planes[Top] = Planef(topNorm * viewSpaceN, m_conners[RTN]);
    m_planes[Bottom] = Planef(botNorm * viewSpaceN, m_conners[LBN]);
    m_planes[Near] = Planef(nearNorm * viewSpaceN, m_conners[LBN]);
    m_planes[Far] = Planef(farNorm * viewSpaceN, m_conners[RTF]);
}

bool Frustum::isInside(const Vec3f &p) const
{
    bool res = true;

    for (int i=0; i<6 && res == true; i++)
    {
        const PlaneSide side = m_planes[i].classifyPoint(p);
        if ( side == Back )
            res = false;
    }

    return res;
}

bool Frustum::isInside(const Aabb &box) const
{
    bool res = true;

    for (int i=0; i<6 && res == true; i++)
    {
        const Vec3f &n(m_planes[i].normal);
        const Vec3f l = box.maxLookUp(n);

        float m = n[0] * l[0] + n[1] * l[1] + n[2] * l[2];

        if ( m < -m_planes[i].d)
            res = false;
    }

    return res;
}

bool Frustum::isInside(const BoundingSphere &sphere) const
{
    bool res = true;
    for (int i=0; i<6 && res == true; i++)
    {
        const Vec3f &n(m_planes[i].normal);
        float d = vecDot(n, sphere.center()) + m_planes[i].d;

        if ( d < -sphere.radius() )
            res = false;
        else if ( std::fabs(d) < sphere.radius() )
            return true;
    }

    return res;
}

void Frustum::getConners(Vec3f v[8]) const
{
    for (int k=0; k<8; k++)
        v[k] = m_conners[k];
}

void Frustum::draw() const
{
    glPushAttrib(GL_ENABLE_BIT | GL_POLYGON_BIT);
    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);

    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glPointSize(4.0);

    glBegin(GL_QUADS);
    glVertex3fv(m_conners[LTN].v);
    glVertex3fv(m_conners[LTF].v);
    glVertex3fv(m_conners[LBF].v);
    glVertex3fv(m_conners[LBN].v);

    glVertex3fv(m_conners[LTN].v);
    glVertex3fv(m_conners[LTF].v);
    glVertex3fv(m_conners[RTF].v);
    glVertex3fv(m_conners[RTN].v);

    glVertex3fv(m_conners[RBN].v);
    glVertex3fv(m_conners[RTN].v);
    glVertex3fv(m_conners[RTF].v);
    glVertex3fv(m_conners[RBF].v);

    glVertex3fv(m_conners[LBN].v);
    glVertex3fv(m_conners[LBF].v);
    glVertex3fv(m_conners[RBF].v);
    glVertex3fv(m_conners[RBN].v);
    glEnd();
    glPopAttrib();
}

void Frustum::extractFromOGLState()
{
    Matrix44f modelview;
    Matrix44f projection;

    glGetFloatv(GL_MODELVIEW_MATRIX, modelview.m);
    glGetFloatv(GL_PROJECTION_MATRIX, projection.m);

    extract(projection * modelview);
}

void Frustum::split(const Matrix44f &viewMatrix,
                    const Projection &proj,
                    const float parts[], size_t nparts,
                    Projection projout[],
                    Frustum frusout[], float epsilon)
{
    proj.split(parts, nparts, projout, epsilon);
    for (size_t i=0; i < nparts; ++i)
    {
        frusout[i] = Frustum(viewMatrix, projout[i]);
    }
}

void Frustum::extractAabb(Aabb *box) const
{
    *box = Aabb();
    for (int k=0; k<8; k++)
    {
        box->add(m_conners[k]);
    }
}

std::ostream& operator<<(std::ostream &out, const Frustum &frus)
{
    out<<"Top: "<<frus.getPlane(Frustum::Top)<<'\n'
       <<"Bottom: "<<frus.getPlane(Frustum::Bottom)<<'\n'
       <<"Left: "<<frus.getPlane(Frustum::Left)<<'\n'
       <<"Right: "<<frus.getPlane(Frustum::Right)<<'\n'
       <<"Near: "<<frus.getPlane(Frustum::Near)<<'\n'
       <<"Far: "<<frus.getPlane(Frustum::Far)<<'\n'

       <<"LBN: "<<frus.getConners()[Frustum::LBN]<<'\n'
       <<"LTN: "<<frus.getConners()[Frustum::LTN]<<'\n'
       <<"LBF: "<<frus.getConners()[Frustum::LBF]<<'\n'
       <<"LTF: "<<frus.getConners()[Frustum::LTF]<<'\n'

       <<"RBN: "<<frus.getConners()[Frustum::RBN]<<'\n'
       <<"RTN: "<<frus.getConners()[Frustum::RTN]<<'\n'
       <<"RBF: "<<frus.getConners()[Frustum::RBF]<<'\n'
       <<"RTF: "<<frus.getConners()[Frustum::RTF];

    return out;
}

UD_NAMESPACE_END
