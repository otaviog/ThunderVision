#ifndef TDV_RECONSTRUCTWINDOW_HPP
#define TDV_RECONSTRUCTWINDOW_HPP

#include <QGLWidget>

class ReconstructWidget: public QGLWidget
{
public:
    ReconstructWidget();
    
private:    
    MeshRenderer *meshRenderer;
};

#endif /* TDV_RECONSTRUCTWINDOW_HPP */
