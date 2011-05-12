#ifndef TDV_TRISTRIPMESH_HPP
#define TDV_TRISTRIPMESH_HPP

#include <tdvbasic/common.hpp>
#include <ud/math/vec3.hpp>
#include "vertexbuffer.hpp"

TDV_NAMESPACE_BEGIN

class TriStripMesh
{
public:        
    TripStripMesh();
        
    void resizeVerts(size_t size)
    {
        if ( m_nVerts != size )
        {
            m_nVerts = size; 
            m_vertices.bind(GL_ARRAY_BUFFER, GL_STATIC_DRAW, size*sizeof(Vec3f));
        }
    }
    
    void resizeIndices(size_t size)
    {
        if ( m_nIndices != size )
        {
            m_nIndices = size;
            m_indices.bind(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW, size*sizeof(GLuint));
        }        
    }
    
    Vec3f* mapVertsRW()
    {
        return m_vertices.map<Vec3f>();
    }     
    
    Vec3f* mapColorsRW()
    {
        return m_colors.map<Vec3f>();
    }
    
    GLuint* mapIndicesRW()
    {
        return m_indices.map<GLuint>();
    }
    
    void draw();
    
private:
    VertexBuffer m_vertices,    
        m_colors,
        m_indices;  
    size_t m_nVerts, m_nIndices;
};

TDV_NAMESPACE_END

#endif /* TDV_MESH_HPP */

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
