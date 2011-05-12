#include "gridglmesh.hpp"

TDV_NAMESPACE_BEGIN

GridGLMesh::GridGLMesh()
    : m_dim(-1)
{
    m_vertBuff = NULL;
    m_colBuff = NULL;
}

void GridGLMesh::resize(const Dim &dim)
{
    if ( m_dim != dim )
    {
        m_dim = dim;
        m_vertices.bind(GL_ARRAY_BUFFER, GL_STATIC_DRAW, 
                        dim.size()*sizeof(Vec3f));
        m_colors.bind(GL_ARRAY_BUFFER, GL_STATIC_DRAW, 
                      dim.size()*sizeof(Vec3f));
        m_indices.bind(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW, 
                       (dim.height()*2*(dim.width() - 1))*sizeof(GLuint));
        
        GLuint *indices = m_indices.map<GLuint>();
        GLuint count = 0;
        for (size_t c=0; c<dim.width() - 1; c++)
        {
            for (size_t r=0; r<dim.height(); r++)
            {
                indices[count++] = r*dim.width() + c;
                indices[count++] = r*dim.width() + c + 1;
            }
        }
        m_indices.unmap();
    }
}

void GridGLMesh::draw()
{
    glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, m_vertices.get());
    glVertexPointer(3, GL_FLOAT, 0, NULL);
    
    glBindBuffer(GL_ARRAY_BUFFER, m_colors.get());
    glColorPointer(3, GL_FLOAT, 0, NULL);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indices.get());
    
    for (size_t col=0; col<m_dim.width() - 1; col++)
    {
        glDrawRangeElements(
            GL_TRIANGLE_STRIP, 0, m_dim.size(),
            m_dim.height()*2, GL_UNSIGNED_INT, 
            (GLvoid*) (col*m_dim.height()*2*sizeof(GLuint)));
    }

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    glPopAttrib();
}

void GridGLMesh::lock()
{
    m_vertBuff = m_vertices.map<Vec3f>();
    m_colBuff = m_colors.map<Vec3f>();
}
    
void GridGLMesh::unlock()
{
    m_vertices.unmap();
    m_colors.unmap();
    m_vertBuff = NULL;
    m_colBuff = NULL;
}

TDV_NAMESPACE_END
