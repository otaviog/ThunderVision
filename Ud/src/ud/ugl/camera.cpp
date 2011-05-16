#include <math.h>
#include "../common.hpp"
#include "../math/matrix44.hpp"
#include "aabb.hpp"
#include "camera.hpp"

UD_NAMESPACE_BEGIN

Camera::Camera()
        : m_eye(0.0, 0.0, 0.0),
        m_view(0.0, 0.0, -1.0),
        m_up(0.0, 1.0, 0.0)
{ }

Camera::Camera(const Vec3f &pos, const Vec3f &view,
               const Vec3f &up)
        : m_eye(pos), m_view(view), m_up(up)
{ }

Camera::Camera(float pos_x, float pos_y, float pos_z,
               float view_x, float view_y, float view_z,
               float up_x, float up_y, float up_z)
        : m_eye(pos_x, pos_y, pos_z),
        m_view(view_x, view_y, view_z),
        m_up(up_x, up_y, up_z)
{ }

void Camera::rotateX(float angle)
{
    Vec3f v(m_view - m_eye);
    Vec3f right(vecNormal(vecCross(v, m_up)));

    Quatf q = rotationQuat(angle, right);
    m_view = q.rotate(v);
    //m_up = vecNormal(vecCross(right, m_view)); // wrong?
    m_view = m_eye + m_view;
}

void Camera::rotateY(float angle)
{
    Vec3f v(m_view - m_eye);
    Vec3f right(vecNormal(vecCross(v, m_up)));

    Quatf q = rotationQuat(angle, m_up);
    m_view = m_eye + q.rotate(v);
}

void Camera::rotate(float xangle, float yangle)
{   
    Vec3f v(m_view - m_eye);
    Vec3f right(vecNormal(vecCross(v, m_up)));
    
    Quatf q = rotationQuat(xangle, right)*rotationQuat(yangle, m_up);
    m_view = m_eye + q.rotate(v);
    m_up = q.rotate(m_up);
}

void Camera::rotateZ(float angle)
{
    Vec3f n(vecNormal(m_view - m_eye));
    Quatf q = rotationQuat(angle, n);
    m_up = q.rotate(m_up);
}

void Camera::rotateEyeX(float angle)
{
    // Fixme
    Vec3f v(m_eye - m_view);
    Vec3f right = vecNormal(vecCross(v, m_up));
    Quatf q = rotationQuat(angle, right);
    Vec3f t = q.rotate(v);

    //m_up = vecNormal(vecCross(right, t));
    m_eye = t + m_view;
}

void Camera::rotateEyeY(float angle)
{
    Vec3f v(m_eye - m_view);
    Quatf q = rotationQuat(angle, m_up);

    v = q.rotate(v);
    m_eye = v + m_view;
}

void Camera::setPosition(const Vec3f &pos)
{
    m_view += pos - m_eye;
    m_eye = pos;
}

void Camera::translate(float xt, float yt, float zt)
{
    m_eye[0] += xt;
    m_eye[1] += yt;
    m_eye[2] += zt;
}

void Camera::move(float ac)
{
    Vec3f v = m_view - m_eye;
    v.normalize();
    m_eye = v * ac + m_eye;
    m_view = v * ac + m_view;
}

void Camera::move(const Vec3f &dir)
{
    m_eye = m_eye + dir;
    m_view = m_view + dir;
}

void Camera::moveXZ(float ac)
{
    Vec3f v = m_view - m_eye;
    v.normalize();
    v[1] = 0.0f;
    move(v*ac);
}

void Camera::setRotation(const Quatf &rot)
{
    m_view = m_eye + rot.rotate(Vec3f(0.0f, 0.0f, -1.0f));
}

void Camera::strife(float ac)
{
    const Vec3f view = vecNormal(m_view - m_eye);
    const Vec3f right = vecCross(view, m_up);

    m_view += right * ac;
    m_eye += right * ac;
}

void Camera::lookAt() const
{
    gluLookAt(m_eye[0], m_eye[1], m_eye[2],
              m_view[0], m_view[1], m_view[2],
              m_up[0], m_up[1], m_up[2]);
}

void Camera::genMatrix(Matrix44f *matrix) const
{
    glPushMatrix();
    glLoadIdentity();
    lookAt();
    glGetFloatv(GL_MODELVIEW_MATRIX, matrix->m);
    glPopMatrix();
}

void Camera::applyGL() const
{
    glLoadIdentity();
    lookAt();
}

void Camera::setAabb(const Aabb &box, const Vec3f &eyeVec, float distance)
{
    const Vec3f center(box.center());
    const float dist = vecLength(box.getMin() - box.getMax())*distance;    
    setView(center);
    setEye(center + eyeVec*dist);
    setUp(vecNormal(
              vecCross(vecNormal(getView() - getEye()),
                       vecCross(eyeVec, ud::Vec3f(0.0f, 1.0f, 0.0f)))));
}

UD_NAMESPACE_END
