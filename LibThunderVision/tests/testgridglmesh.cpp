#include <tdvision/gridglmesh.hpp>
#include <cstdlib>
#include <GL/glut.h>

static const float NEAR = 1.0f;
static const float FAR = 50.0f;

tdv::GridGLMesh *mesh;

void keyboard(unsigned char key, int x, int y)
{
    if ( key == 27 )
    {
        exit(0);
    }
}

void reshape(int w, int h)
{
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-1, 1, -1, 1, NEAR, FAR);
    glMatrixMode(GL_MODELVIEW);
}

void display()
{
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -1.0f);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    mesh->draw();
    
    glutSwapBuffers();
}

int main(int argc, char *argv[])
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE);
    
    glutCreateWindow("Test GridGLMesh");
    glutKeyboardFunc(keyboard);
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    
    glewInit();
    
    mesh = new tdv::GridGLMesh;
    mesh->resize(tdv::Dim(512, 512));
    
    mesh->lock();
    
    for (size_t i=0; i<512; i++)
    {
        for (size_t j=0; j<512; j++)
        {
            const float st = float(i + j)/1024.0f;
            mesh->setPoint(
                j, i,
                ud::Vec3f(
                    float(j)/512.0f*2.0f - 1.0f, 
                    float(i)/512.0f*2.0f - 1.0f, 
                    -(NEAR + st)),
                ud::Vec3f(st, 0.0f, 1.0f-st));
        }
    }
    
    mesh->unlock();
    
    glutMainLoop();

    return 0;
}
