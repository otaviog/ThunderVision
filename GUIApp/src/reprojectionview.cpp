#include "reprojectionview.hpp"

ReprojectionView::ReprojectionView(QWidget *parent)
    : QGLWidget(parent)
{
    m_reproj = NULL;
}

void ReprojectionView::paintGL()
{
    glLoadIdentity();            
    
    if ( m_reproj != NULL )
    {
        const ud::Aabb &box = m_reproj->box();
        const ud::Vec3f center(box.center());
        const ud::Vec3f &mi = box.getMin();
        const ud::Vec3f &ma = box.getMax();
        
        glTranslatef(-(ma[0] - mi[0])*0.5f, 
                     (ma[1] - mi[1])*0.5f, -5.0f);
        
        m_reproj->draw();        
    }
}
    
void ReprojectionView::resizeGL(int width, int height)
{
    const float aspect = (width < height) 
        ? float(width)/float(height) 
        : float(height)/float(width);
    
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, aspect, 1.0f, 100.0f);    
    glMatrixMode(GL_MODELVIEW);    
}

void ReprojectionView::initializeGL()
{
    glewInit();
}
