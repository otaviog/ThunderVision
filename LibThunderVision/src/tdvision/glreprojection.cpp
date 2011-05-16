#include "sink.hpp"
#include "glreprojection.hpp"

TDV_NAMESPACE_BEGIN

GLReprojection::GLReprojection()
{
    m_lorigin = NULL;
    m_lrepr = NULL;
}

void GLReprojection::updateMesh()
{    
    if ( m_lorigin == NULL )
        return ;
    
    const Dim &dim = m_ldisp.dim();
    
    m_mesh.resize(dim);
    m_mesh.lock();
    
    const float *disp_c = m_ldisp.cpuMem()->data.fl;   
    const size_t step = m_lorigin->step;
    const float _1_255 = 1.0f/255.0f;
    
    m_box.clear();
    
    for (size_t r=0; r<dim.height(); r++)
    {
        for (size_t c=0; c<dim.width(); c++)
        {
            uchar *colorBase = &m_lorigin->data.ptr[r*step + c];

            const ud::Vec3f color(colorBase[0]*_1_255,
                                  colorBase[1]*_1_255,
                                  colorBase[2]*_1_255);
            const ud::Vec3f vert(
                m_lrepr->reproject(
                    c, r, disp_c[r*dim.width() + c]));
            
            m_box.add(vert);
            
            m_mesh.setPoint(
                c, r, vert,
                color);
        }
    }
    
    m_mesh.unlock();
    
    m_lorigin = NULL;
}

void GLReprojection::reproject(FloatImage disp, CvMat *origin, 
                               Reprojector *repr)
{
    assert(repr != NULL);    
    boost::mutex::scoped_lock lock(m_meshMutex);
    
    m_ldisp = disp;
    m_lorigin = origin;
    m_lrepr = repr;
    
    CvMatSinkPol::incrRef(m_lorigin);
}

void GLReprojection::draw()
{
    boost::mutex::scoped_lock lock(m_meshMutex);    
    updateMesh();        
    
    CvMatSinkPol::sink(m_lorigin);
    
    m_mesh.draw();        
}

TDV_NAMESPACE_END

