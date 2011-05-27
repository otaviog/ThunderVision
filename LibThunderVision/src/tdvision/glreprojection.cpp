#include "sink.hpp"
#include "glreprojection.hpp"

TDV_NAMESPACE_BEGIN

GLReprojection::GLReprojection()
{
    m_lorigin = NULL;
    m_lrepr = NULL;
    m_observer = NULL;
}

void GLReprojection::updateMesh()
{    
    if ( m_lorigin == NULL )
        return ;
    
    const Dim &dim = m_ldisp.dim();
    
    m_mesh.resize(dim);
    m_mesh.lock();

#if 1    
    const float *disp_c = m_ldisp.cpuMem()->data.fl;   
    const float _1_255 = 1.0f/255.0f;
    
    m_box.clear();
    
    for (size_t r=0; r<dim.height(); r++)
    {
        const float *dispPtr = disp_c + r*dim.width();
        const uchar *colorPtr = m_lorigin->data.ptr + r*m_lorigin->step;
        

        for (size_t c=0; c<dim.width(); c++)
        {
            const uchar *colorBase = colorPtr + c*3;
            
            const ud::Vec3f color(colorBase[0]*_1_255,
                                  colorBase[1]*_1_255,
                                  colorBase[2]*_1_255);
            const ud::Vec3f vert(
                m_lrepr->reproject(
                    c, r, *dispPtr, dim));
            
            m_box.add(vert);
            
            m_mesh.setPoint(
                c, r, vert,
                color);
            
            dispPtr++;
        }
    }        
#else
    RectificationCV *cv = dynamic_cast<RectificationCV*>(m_lrepr);
    CvMat *Q = cvMat(4, 4, CV_32F, cv->reprojectionMatrix());
    
#endif
    
    m_mesh.unlock();
    
    m_lorigin = NULL;
}

void GLReprojection::reproject(FloatImage disp, CvMat *origin, 
                               Reprojector *repr)
{
    assert(repr != NULL);
    
    {
        boost::mutex::scoped_lock lock(m_meshMutex);
    
        m_ldisp = disp;
        m_lorigin = origin;
        m_lrepr = repr;
        
        CvMatSinkPol::incrRef(m_lorigin);
    }
    
    if ( m_observer != NULL )
        m_observer->reprojectionUpdated();
}

void GLReprojection::draw()
{
    boost::mutex::scoped_lock lock(m_meshMutex);    
    updateMesh();        
    
    CvMatSinkPol::sink(m_lorigin);
    
    m_mesh.draw();        
}

TDV_NAMESPACE_END

