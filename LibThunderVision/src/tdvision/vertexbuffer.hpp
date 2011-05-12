#ifndef TDV_VERTEXBUFFER_HPP
#define TDV_VERTEXBUFFER_HPP

#include <tdvbasic/common.hpp>
#include "glcommon.hpp"

TDV_NAMESPACE_BEGIN

class VertexBuffer
{
public:
    VertexBuffer()
    {
        glGenBuffers(1, &m_id);
        m_elemType = GL_NONE;
        m_mode = GL_NONE;
        m_size = 0;
    }

    ~VertexBuffer()
    {
        glDeleteBuffers(1, &m_id);
    }
    

    VertexBuffer(const VertexBuffer &cpy)
    {
        glGenBuffers(1, &m_id);
        copy(cpy);
    }

    VertexBuffer& operator=(const VertexBuffer &cpy)
    {
        copy(cpy);
        return *this;
    }    
    
    void bind(GLenum type, GLenum mode, size_t size,
              void *data = NULL);
        
    template<typename Ptr>
    const Ptr* mapReadOnly() const
    {
        if ( m_elemType == GL_NONE )
            return NULL;
        
        glBindBuffer(m_elemType, m_id);
        return reinterpret_cast<Ptr*>(glMapBuffer(m_elemType, GL_READ_ONLY));
    }
    
    template<typename Ptr>
    Ptr* mapWriteOnly()
    {
        if ( m_elemType == GL_NONE )
            return NULL;
        
        glBindBuffer(m_elemType, m_id);
        return reinterpret_cast<Ptr*>(glMapBuffer(m_elemType, GL_WRITE_ONLY));
    }
    
    template<typename Ptr>
    Ptr* map()
    {
        if ( m_elemType == GL_NONE )
            return NULL;
        glBindBuffer(m_elemType, m_id);
        return reinterpret_cast<Ptr*>(glMapBuffer(m_elemType, GL_READ_WRITE));
    }

    void unmap()
    {
        glBindBuffer(m_elemType, m_id);
        glUnmapBuffer(m_elemType);
    }
    
    GLuint get()
    {
        return m_id;
    }

private:
    
    void copy(const VertexBuffer &cpy);
    
    GLuint m_id;
    GLenum m_elemType, m_mode;
    size_t m_size;    
};

TDV_NAMESPACE_END

#endif /* TDV_VERTEXBUFFER_HPP */
