#include <string>
#include "../common.hpp"

UD_NAMESPACE_BEGIN

class Planef;
class Mesh;

namespace misc
{
    void DrawPlane(const Planef &plane, GLfloat planeScale) throw();
    void ShowTexture(GLuint tex) throw();

    void ShowTexture(GLenum target, GLuint tex, float startX, float startY,
                     float endX, float endY, int width, int height) throw();
    
    inline void ShowRectTexture(GLuint tex, float startX, float startY,
                                float endX, float endY, int width, int height) throw()
    {
        ShowTexture(GL_TEXTURE_RECTANGLE_ARB, tex, startX, startY, endX, endY,
                    width, height);
                        
    }

    inline void ShowTexture(GLuint tex, float startX, float startY,
                            float endX, float endY) throw()
    {
        ShowTexture(GL_TEXTURE_2D, tex, startX, startY, endX, endY, 1, 1);
    }
    
    void meshtoOBJ(const Mesh &mesh, const std::string &outfilename);    
}

UD_NAMESPACE_END
