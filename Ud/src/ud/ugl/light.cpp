#include "../math/matrix44.hpp"
#include "light.hpp"

UD_NAMESPACE_BEGIN

void Light::genViewMatrix(Matrix44f *pViewMtx) const
{
    const Vec3f view(0.0f, 0.0f, 0.0f);
    const Vec3f vp = view - vec_cast<Vec3f>(position);
    const Vec3f up = vecCross(
        vecCross(vp, Vec3f(0.0f, 1.0f, 0.0f)), vp);

    glPushMatrix();
    glLoadIdentity();
    gluLookAt(position[0], position[1], position[2],
              0, 0, 0,
              0, 1, 0);
    glGetFloatv(GL_MODELVIEW_MATRIX, pViewMtx->m);
    glPopMatrix();
}

void Light::draw(float radius) const
{
    GLUquadric *quad = gluNewQuadric();    
    glPushMatrix();
    glTranslatef(position[0], position[1], position[2]);
    gluSphere(quad, radius, 8, 8);
    glPopMatrix();

    gluDeleteQuadric(quad);
}

void Light::applyGL(GLenum light) const
{
    glLightfv(light, GL_AMBIENT, ambient.array());
    glLightfv(light, GL_DIFFUSE, diffuse.array());
    glLightfv(light, GL_SPECULAR, specular.array());
    glLightfv(light, GL_POSITION, position.v);
}

UD_NAMESPACE_END
