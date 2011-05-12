#ifndef TDV_REPROJECTIONVIEW_HPP
#define TDV_REPROJECTIONVIEW_HPP

#include <tdvbasic/common.hpp>
#include <tdvision/glreprojection.hpp>
#include <QGLWidget>

class ReprojectionView: public QGLWidget
{
    Q_OBJECT;
public:
    ReprojectionView(QWidget *parent = NULL);
    
    tdv::GLReprojection* reprojection()
    {
        if ( m_reproj == NULL )
        {
            m_reproj = new tdv::GLReprojection;
        }
        
        return m_reproj;        
    }
    
protected:
    virtual void paintGL();
    
    virtual void resizeGL(int width, int height);
    
    void initializeGL();
    
private:
    tdv::GLReprojection *m_reproj;
};

#endif /* TDV_REPROJECTIONVIEW_HPP */
