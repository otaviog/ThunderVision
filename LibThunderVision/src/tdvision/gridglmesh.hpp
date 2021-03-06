#ifndef TDV_GRIDGLMESH_HPP
#define TDV_GRIDGLMESH_HPP

#include <tdvbasic/common.hpp>
#include <ud/math/vec3.hpp>
#include "dim.hpp"
#include "vertexbuffer.hpp"

TDV_NAMESPACE_BEGIN

class GridGLMesh
{
public:        
    GridGLMesh();
        
    void resize(const Dim &dim);
    
    void setPoint(int x, int y, const ud::Vec3f &vert, const ud::Vec3f &color)
    {
        assert(m_vertBuff != NULL);
        
        const size_t idx = y*m_dim.width() + x;
        m_vertBuff[idx] = vert;
        m_colBuff[idx] = color;
    }

    void point(int x, int y, ud::Vec3f &vert, ud::Vec3f &color) const
    {
        assert(m_vertBuff != NULL);
        
        const size_t idx = y*m_dim.width() + x;
        vert = m_vertBuff[idx];
        color = m_colBuff[idx];
    }
  
    void lock();
    
    void unlock();
    
    void draw();

    
    const Dim& dim() const
    {
        return m_dim;
    }

private:
    VertexBuffer m_vertices,    
        m_colors,
        m_indices;  
    Dim m_dim;
    
    ud::Vec3f *m_vertBuff, *m_colBuff;    
};

class ScopedMeshLock
{
public:
    ScopedMeshLock(GridGLMesh &mesh)
        : m_mesh(mesh)
    {
        m_mesh.lock();
    }
    
    ~ScopedMeshLock()
    {
        m_mesh.unlock();
    }

private:
    GridGLMesh &m_mesh;    
};

TDV_NAMESPACE_END

#endif /* TDV_GRIDGLMESH_HPP */
