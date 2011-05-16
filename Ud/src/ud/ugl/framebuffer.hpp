#ifndef UD_FRAMEBUFFER_HPP
#define UD_FRAMEBUFFER_HPP

#include <vector>
#include "../common.hpp"
#include "../noncopyable.hpp"

UD_NAMESPACE_BEGIN

class Renderbuffer: public NonCopyable
{
public:
    Renderbuffer();
    ~Renderbuffer();
    
    void storage(GLenum component, int width, int height);

    GLuint oglName()
    {
        return m_rb;
    }
    
private:
    GLuint m_rb;
};

class Framebuffer: public NonCopyable
{
public:
    Framebuffer(int width, int height);
    ~Framebuffer();
    
    void attachColorTarget(GLenum attachment, GLenum format, GLenum target, GLuint tex);
    void attachColorTarget(GLenum attachment, Renderbuffer &target);
    void attachRenderbuffer(GLenum attachment, GLenum format);

    void setDepthTarget(GLuint tex);
    
    void size(int width, int height);
    void sizeFromViewport();
    
    void begin();
    void begin(GLenum attachment);
    
    void beginDepthmap();
    
    void end();
    
    void enableDepth(bool enable);
    
    bool isDepthEnabled() const
    {
        return m_depthEnabled;
    }
    
    int width() const
    {
        return m_width;
    }
    
    int height() const
    {
        return m_height;
    }
    
private:
    void attachColorTargetImpl(GLenum attachment, GLenum format, GLenum target, GLuint tex);
    void setDepthTargetImpl(GLuint tex);
    void attachRenderbufferImpl(GLenum attachment, GLenum component, 
                                GLuint renderbuffer);
    
    struct ColorAttachInfo 
    {
        ColorAttachInfo(GLenum a, GLenum f, GLenum t, GLuint name)
        {
            attachment = a;
            format = f;
            target = t;
            oglName = name;
        }
        
        GLenum attachment, format, target;
        GLuint oglName;
    };
    
    std::vector<ColorAttachInfo> m_texColorAttachs;
    std::vector<ColorAttachInfo> m_rbColorAttachs;
    
    GLuint m_fbo, m_depthTarget;
    Renderbuffer m_depthbuffer;
    
    bool m_depthEnabled;
    int m_width, m_height;
};

UD_NAMESPACE_END

#endif /* UD_FRAMEBUFFER_HPP */
