#include <fstream>
#include "../math/plane.hpp"
#include "misc.hpp"

UD_NAMESPACE_BEGIN

namespace misc
{
    /* http://www.gamedev.net/community/forums/topic.asp?topic_id=340803 */
    void DrawPlane(const Planef &plane, GLfloat planeScale) throw()
    {

        glPushAttrib(GL_LIGHTING_BIT | GL_TEXTURE_BIT
                     | GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
        glDisable(GL_LIGHTING);
        glDisable(GL_TEXTURE_2D);
        glDepthMask(GL_FALSE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        const Vec3f p0 = -plane.normal * plane.d;

        Vec3f u(plane.normal[1] - plane.normal[2],
                plane.normal[2] - plane.normal[0],
                plane.normal[0] - plane.normal[1]);
        Vec3f v(vecCross(plane.normal, u));

        u *= planeScale;
        v *= planeScale;

        const Vec3f pts[4] =
            {
                p0 - u - v,
                p0 + u - v,
                p0 + u + v,
                p0 - u + v
            };

        glColor4f(0.0f, 0.0f, 1.0f, 0.5f);

        glBegin(GL_QUADS);
        glVertex3fv(pts[0].v);
        glVertex3fv(pts[1].v);
        glVertex3fv(pts[2].v);
        glVertex3fv(pts[3].v);
        glEnd();

        glPopAttrib();
    }

    void ShowTexture(GLenum target, GLuint tex, float startX, float startY,
                     float endX, float endY, int width, int height) throw()
    {        
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(-1.0f, 1.0f, -1.0f, 1.0f, -1, 0);

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        glPushAttrib(GL_LIGHTING_BIT | GL_TEXTURE_BIT | GL_DEPTH_BUFFER_BIT);
        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_TEXTURE_2D);
        glEnable(target);
        glActiveTexture(GL_TEXTURE0);
        
        glBindTexture(target, tex);
        glColor3f(1.0f, 1.0f, 1.0f);
        
        glBegin(GL_QUADS);
        glTexCoord2i(0, height); glVertex2f(startX, startY);
        glTexCoord2i(0, 0); glVertex2f(startX, endY);
        glTexCoord2i(width, 0); glVertex2f(endX, endY);
        glTexCoord2i(width, height); glVertex2f(endX, startY);
        glEnd();
        
        glPopAttrib();
        
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();

        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
    }   
}

UD_NAMESPACE_END
