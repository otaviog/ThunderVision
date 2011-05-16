#include "aabb.hpp"

UD_NAMESPACE_BEGIN

void Aabb::draw() const
{
    glPushAttrib(GL_LIGHTING_BIT | GL_POLYGON_BIT);
    glDisable(GL_LIGHTING);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_QUADS);

    glVertex3f(m_min[0], m_min[1], m_min[2]);
    glVertex3f(m_max[0], m_min[1], m_min[2]);
    glVertex3f(m_max[0], m_max[1], m_min[2]);
    glVertex3f(m_min[0], m_max[1], m_min[2]);

    glVertex3f(m_min[0], m_min[1], m_max[2]);
    glVertex3f(m_max[0], m_min[1], m_max[2]);
    glVertex3f(m_max[0], m_max[1], m_max[2]);
    glVertex3f(m_min[0], m_max[1], m_max[2]);

    glVertex3f(m_min[0], m_max[1], m_max[2]);
    glVertex3f(m_min[0], m_max[1], m_min[2]);
    glVertex3f(m_min[0], m_min[1], m_min[2]);
    glVertex3f(m_min[0], m_min[1], m_max[2]);

    glVertex3f(m_max[0], m_max[1], m_max[2]);
    glVertex3f(m_max[0], m_max[1], m_min[2]);
    glVertex3f(m_max[0], m_min[1], m_min[2]);
    glVertex3f(m_max[0], m_min[1], m_max[2]);

    glEnd();
    glPopAttrib();

}

// 3D Math Primer for Graphics and Game Development
void Aabb::transform(const Matrix44f &mtx)
{
    const Vec3f bmin(m_min), bmax(m_max);

    if ( mtx[0] > 0.0f )
    {
        m_min[0] = mtx[0] * bmin[0];
        m_max[0] = mtx[0] * bmax[0];
    }
    else
    {
        m_min[0] = mtx[0] * bmax[0];
        m_max[0] = mtx[0] * bmin[0];
    }

    if ( mtx[4] > 0.0f )
    {
        m_min[0] += mtx[4] * bmin[1];
        m_max[0] += mtx[4] * bmax[1];
    }
    else
    {
        m_min[0] += mtx[4] * bmax[1];
        m_max[0] += mtx[4] * bmin[1];
    }

    if ( mtx[8] > 0.0f )
    {
        m_min[0] += mtx[8] * bmin[2];
        m_max[0] += mtx[8] * bmax[2];
    }
    else
    {
        m_min[0] += mtx[8] * bmax[2];
        m_max[0] += mtx[8] * bmin[2];
    }

    if ( mtx[1] > 0.0f )
    {
        m_min[1] = mtx[1] * bmin[0];
        m_max[1] = mtx[1] * bmax[0];
    }
    else
    {
        m_min[1] = mtx[1] * bmax[0];
        m_max[1] = mtx[1] * bmin[0];
    }

    if ( mtx[5] > 0.0f )
    {
        m_min[1] += mtx[5] * bmin[1];
        m_max[1] += mtx[5] * bmax[1];
    }
    else
    {
        m_min[1] += mtx[5] * bmax[1];
        m_max[1] += mtx[5] * bmin[1];
    }

    if ( mtx[9] > 0.0f )
    {
        m_min[1] += mtx[9] * bmin[2];
        m_max[1] += mtx[9] * bmax[2];
    }
    else
    {
        m_min[1] += mtx[9] * bmax[2];
        m_max[1] += mtx[9] * bmin[2];
    }

    if ( mtx[2] > 0.0f )
    {
        m_min[2] = mtx[2] * bmin[0];
        m_max[2] = mtx[2] * bmax[0];
    }
    else
    {
        m_min[2] = mtx[2] * bmax[0];
        m_max[2] = mtx[2] * bmin[0];
    }

    if ( mtx[6] > 0.0f )
    {
        m_min[2] += mtx[6] * bmin[1];
        m_max[2] += mtx[6] * bmax[1];
    }
    else
    {
        m_min[2] += mtx[6] * bmax[1];
        m_max[2] += mtx[6] * bmin[1];
    }

    if ( mtx[10] > 0.0f )
    {
        m_min[2] += mtx[10] * bmin[2];
        m_max[2] += mtx[10] * bmax[2];
    }
    else
    {
        m_min[2] += mtx[10] * bmax[2];
        m_max[2] += mtx[10] * bmin[2];
    }

    m_min[0] += mtx[12];
    m_min[1] += mtx[13];
    m_min[2] += mtx[14];

    m_max[0] += mtx[12];
    m_max[1] += mtx[13];
    m_max[2] += mtx[14];
}

void Aabb::extractPoints(Vec3f points[8]) const
{
    points[0] = Vec3f(m_min[0], m_max[1], m_max[2]);
    points[1] = Vec3f(m_min[0], m_min[1], m_max[2]);
    points[2] = Vec3f(m_max[0], m_min[1], m_max[2]);
    points[3] = Vec3f(m_max[0], m_max[1], m_max[2]);
    
    points[4] = Vec3f(m_min[0], m_max[1], m_min[2]);
    points[5] = Vec3f(m_min[0], m_min[1], m_min[2]);
    points[6] = Vec3f(m_max[0], m_min[1], m_min[2]);
    points[7] = Vec3f(m_max[0], m_max[1], m_min[2]);
}

Aabb Aabb::splitZ(float startU, float endU) const
{
    Vec3f nmin, nmax;
    
    nmin = vecLinearInterpolate(m_min, Vec3f(m_min[0], m_min[1], m_max[2]), startU);
    nmax = vecLinearInterpolate(Vec3f(m_max[0], m_max[1], m_min[2]), m_max, endU);
    
    return Aabb(nmin, nmax);
}

UD_NAMESPACE_END
