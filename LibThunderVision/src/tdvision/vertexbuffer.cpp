#include "vertexbuffer.hpp"

TDV_NAMESPACE_BEGIN

void VertexBuffer::bind(GLenum type, GLenum mode, size_t size,
                        void *data) 
{
    m_elemType = type;
    glBindBuffer(type, m_id);
    glBufferData(type, size, data, mode);
    glBindBuffer(type, 0);
}

TDV_NAMESPACE_END
