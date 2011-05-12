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
        m_reproj->draw();
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
