#ifndef TDV_GRIDGLMESH_HPP
#define TDV_GRIDGLMESH_HPP

#include <tdvbasic/common.hpp>
#include "vec3.hpp"
#include "dim.hpp"
#include "vertexbuffer.hpp"

TDV_NAMESPACE_BEGIN

class GridGLMesh
{
public:        
    GridGLMesh();
        
    void resize(const Dim &dim);
    
    void setPoint(int x, int y, const Vec3f &vert, const Vec3f &color)
    {
        assert(m_vertBuff != NULL);
        
        const size_t idx = y*m_dim.width() + x;
        m_vertBuff[idx] = vert;
        m_colBuff[idx] = color;
    }
  
    void lock();
    
    void unlock();
    
    void draw();
    
private:
    VertexBuffer m_vertices,    
        m_colors,
        m_indices;  
    Dim m_dim;
    
    Vec3f *m_vertBuff, *m_colBuff;    
};

TDV_NAMESPACE_END

#endif /* TDV_GRIDGLMESH_HPP */

#if 0
class Reconstruction
{
    virtual void reconstruct(FloatImage image, Reprojector *reproj) = 0;

};

class GLReconstruction
{
public:
    
    virtual void draw() = 0;
        
};

class GLMeshReconstruction: Reconstruction
{
    
};

class GLVoxelReconstruction: Reconstruction
{
};

#endif
