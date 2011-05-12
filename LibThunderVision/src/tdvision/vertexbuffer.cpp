#include "vertexbuffer.hpp"

TDV_NAMESPACE_BEGIN

void VertexBuffer::copy(const VertexBuffer &cpy)
{
    m_elemType = cpy.m_elemType;
    m_size = cpy.m_size;
    m_mode = cpy.m_mode;

    if ( m_elemType != GL_NONE )
    {        
        glBindBuffer(m_elemType, m_id);
        
        const unsigned char *data = cpy.mapReadOnly<unsigned char>();
        
        glBufferData(m_elemType, m_size, (GLvoid*) data, m_mode);
        
        glBindBuffer(m_elemType, 0);
    }
}

void VertexBuffer::bind(GLenum type, GLenum mode, size_t size,
                        void *data) 
{
    m_size = size;
    m_elemType = type;
    m_mode = mode;
    
    glBindBuffer(type, m_id);
    glBufferData(type, size, data, mode);
    glBindBuffer(type, 0);
}

TDV_NAMESPACE_END
