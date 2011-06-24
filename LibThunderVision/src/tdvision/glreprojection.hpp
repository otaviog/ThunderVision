#ifndef TDV_GLREPROJECTION_HPP
#define TDV_GLREPROJECTION_HPP

#include <tdvbasic/common.hpp>
#include <ud/ugl/aabb.hpp>
#include <boost/thread.hpp>
#include "reprojection.hpp"
#include "reprojector.hpp"
#include "gridglmesh.hpp"

TDV_NAMESPACE_BEGIN

class MeshExporter;

class GLReprojectionObserver
{
public:
    virtual void reprojectionUpdated() = 0;
    
private:
};

class GLReprojection: public Reprojection
{
public:
    GLReprojection();    

    void reproject(FloatImage image, CvMat *origin, Reprojector *repr);
    
    void draw();    
    
    void exportMesh(MeshExporter *exporter);        
                
    void observer(GLReprojectionObserver *observer)
    {
        m_observer = observer;
    }
    
    const ud::Aabb& box() const
    {
        return m_box;
    }
    
private:            
    void updateMesh();
    
    GridGLMesh m_mesh;
    boost::mutex m_meshMutex;
        
    FloatImage m_ldisp;
    CvMat *m_lorigin;
    Reprojector *m_lrepr;
    
    ud::Aabb m_box;
    
    GLReprojectionObserver *m_observer;
};

TDV_NAMESPACE_END

#endif /* TDV_GLREPROJECT_HPP */
