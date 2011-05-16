#include <cmath>
#include "../math/math.hpp"
#include "../math/vec3.hpp"
#include "../math/matrix44.hpp"
#include "aabb.hpp"
#include "projection.hpp"

UD_NAMESPACE_BEGIN

// Taken from Mesa sources, src/glu/sgi/libuti/project.c
Projection::Projection(float fov, float aspect, float _near, float _far)
{
    float e = 1.0f / std::tan(Math::degToRad(fov/2.0f));
    m_left = -_near/e;
    m_right = _near/e;
    m_top = (aspect * _near)/e;
    m_bottom = -(aspect * _near)/e;

    m_farLeft = -_far/e;
    m_farRight = _far/e;
    m_farTop = _far/e;
    m_farBottom = -_far/e;

    m_nearD = _near;
    m_farD = _far;

    m_fov = fov;
    m_aspect = aspect;
}

Projection::Projection(float _left, float _right,
                       float _bot, float _top,
                       float _near, float _far)
{
    m_left = _left;
    m_right = _right;
    m_bottom = _bot;
    m_top = _top;
    m_nearD = _near;
    m_farD = _far;

    m_farLeft = (m_left/m_nearD)*m_farD;
    m_farRight = (m_right/m_nearD)*m_farD;
    m_farTop = (m_top/m_nearD)*m_farD;
    m_farBottom = (m_bottom/m_nearD)*m_farD;

    m_fov = Math::radToDeg(std::atan(m_right/m_nearD))*2;
    m_aspect = (m_right + std::abs(m_left)) / (m_top + std::abs(m_bottom));
}

void Projection::extractCorners(Vec3f nearConners[4], Vec3f farConners[4]) const
{
    const float lnear = -m_nearD;
    const float lfar = -m_farD;

    nearConners[0] = Vec3f(m_left, m_top, lnear);
    nearConners[1] = Vec3f(m_left, m_bottom, lnear);
    nearConners[2] = Vec3f(m_right, m_bottom, lnear);
    nearConners[3] = Vec3f(m_right, m_top, lnear);

    farConners[0] = Vec3f(m_farLeft, m_farTop, lfar);
    farConners[1] = Vec3f(m_farLeft, m_farBottom, lfar);
    farConners[2] = Vec3f(m_farRight, m_farBottom, lfar);
    farConners[3] = Vec3f(m_farRight, m_farTop, lfar);
}

void Projection::extractAabb(Aabb *box) const
{
    box->setMin(Vec3f(m_left, m_bottom, -m_farD));
    box->setMax(Vec3f(m_farRight, m_farTop, -m_nearD));
}

void Projection::genMatrix(Matrix44f *pMtx) const
{
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glFrustum(m_left, m_right, m_bottom, m_top, m_nearD, m_farD);
    glGetFloatv(GL_PROJECTION_MATRIX, pMtx->m);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

void Projection::applyGL() const
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(m_left, m_right, m_bottom, m_top, m_nearD, m_farD);
    glMatrixMode(GL_MODELVIEW);
}

void Projection::split(const float parts[], size_t nparts, Projection out[],
    float epsilon) const
{
    const float Dfn = m_farD - m_nearD;
    float curNear = m_nearD;

    for (size_t i=0; i<nparts; i++)
    {
        const float curFar = curNear + Dfn * parts[i];
        out[i] = Projection(
            m_fov, m_aspect, curNear-epsilon, curFar+epsilon);
        curNear = curFar;
    }
}

UD_NAMESPACE_END
