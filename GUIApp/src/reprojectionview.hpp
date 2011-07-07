#ifndef TDV_REPROJECTIONVIEW_HPP
#define TDV_REPROJECTIONVIEW_HPP

#include <tdvbasic/common.hpp>
#include <tdvision/glreprojection.hpp>
#include <ud/ugl/camera.hpp>
#include <QGLWidget>

class ReprojectionView: public QGLWidget, public tdv::GLReprojectionObserver
{
    Q_OBJECT;
public:
    ReprojectionView(QWidget *parent = NULL);

    void reprojectionUpdated()
    {
        update();
        m_empty = false;
    }
    
    tdv::GLReprojection* reprojection()
    {
        if ( m_reproj == NULL )
        {
            m_reproj = new tdv::GLReprojection;
            m_reproj->observer(this);
        }
        
        return m_reproj;        
    }
    
public slots:
    void exportMesh();
        
protected:
    virtual void paintGL();
    
    virtual void resizeGL(int width, int height);
    
    virtual void keyPressEvent(QKeyEvent *event);
    
    virtual void mouseMoveEvent(QMouseEvent *event);

    virtual void mousePressEvent(QMouseEvent *event);
    
    virtual void mouseReleaseEvent(QMouseEvent *event);
    
    void initializeGL();
    
private:
    tdv::GLReprojection *m_reproj;
    ud::Camera m_camera;
    ud::Vec2f m_lastMousePos;    
    float m_zcenter;
    bool m_btPressed;
    
    bool m_empty;
};

#endif /* TDV_REPROJECTIONVIEW_HPP */
