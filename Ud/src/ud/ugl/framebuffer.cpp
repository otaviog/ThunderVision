#include "../debug.hpp"
#include "framebuffer.hpp"

UD_NAMESPACE_BEGIN


Renderbuffer::Renderbuffer()
{
    glGenRenderbuffersEXT(1, &m_rb);
}

Renderbuffer::~Renderbuffer()
{
    glDeleteRenderbuffersEXT(1, &m_rb);
}

void Renderbuffer::storage(GLenum component, int width, int height)
{    
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, m_rb);
    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, component, width, height);
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
}

//////////////////////////////////////////////////////////////////////
///////////////// Framebuffer
////////////////////////////////////////////////////////////////////
//////////////// Framebuffer
///////////////////////////////////////////////////////////////////
Framebuffer::Framebuffer(int width, int height)
{
    m_width = width;
    m_height = height;
    
    glGenFramebuffersEXT(1, &m_fbo);
    m_depthbuffer.storage(GL_DEPTH_COMPONENT, m_width, m_height);
}

Framebuffer::~Framebuffer()
{
    for (std::vector<ColorAttachInfo>::iterator it=m_rbColorAttachs.begin(); 
         it != m_rbColorAttachs.end(); ++it)
    {
        const ColorAttachInfo &attachInfo(*it);
        glDeleteRenderbuffersEXT(1, &attachInfo.oglName);
    }

    glDeleteFramebuffersEXT(1, &m_fbo);    
}
    
void Framebuffer::attachColorTargetImpl(GLenum attachment, GLenum format, 
                                        GLenum target, GLuint tex)
{   
    glBindTexture(target, tex);
    glTexImage2D(target, 0, format, m_width, m_height, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, NULL);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, attachment,
                              target, tex, 0);
    glBindTexture(target, 0);
}

void Framebuffer::attachColorTarget(GLenum attachment, GLenum format, 
                                    GLenum target, GLuint tex)
{
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_fbo);
    attachColorTargetImpl(attachment, format, target, tex);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);   
    m_texColorAttachs.push_back(ColorAttachInfo(attachment, format, target, tex));
}

void Framebuffer::attachColorTarget(GLenum attachment, Renderbuffer &target)
{
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_fbo);
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, attachment,
                                 GL_RENDERBUFFER_EXT, target.oglName());
                                 
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);   
}

void Framebuffer::attachRenderbufferImpl(GLenum attachment, GLenum component, 
                                         GLuint renderbuffer)
{
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, renderbuffer); 
    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, component, m_width, m_height);

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_fbo);
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, attachment,
                                 GL_RENDERBUFFER_EXT, renderbuffer);

    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}

void Framebuffer::attachRenderbuffer(GLenum attachment, GLenum component)
{
    GLuint renderbuffer;
    glGenRenderbuffersEXT(1, &renderbuffer);
    attachRenderbufferImpl(attachment, component, renderbuffer);    
    m_rbColorAttachs.push_back(ColorAttachInfo(attachment, component, 
                                               GL_NONE, renderbuffer));
}

void Framebuffer::setDepthTargetImpl(GLuint tex)
{
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, m_width, m_height, 0,
                 GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL);

    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,
                              GL_TEXTURE_2D, tex, 0);
}

void Framebuffer::setDepthTarget(GLuint tex)
{
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_fbo);
    setDepthTargetImpl(tex);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    
    m_depthTarget = tex;
}

void Framebuffer::size(int width, int height)
{
    if ( width == m_width && height == m_height )
        return;
    
    m_width = width;
    m_height = height;
    
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_fbo);
    for (std::vector<ColorAttachInfo>::iterator it=m_texColorAttachs.begin(); 
         it != m_texColorAttachs.end(); ++it)
    {
        const ColorAttachInfo &attachInfo(*it);
        attachColorTargetImpl(attachInfo.attachment, attachInfo.format, 
                              attachInfo.target, attachInfo.oglName);
    }
    
    for (std::vector<ColorAttachInfo>::iterator it=m_rbColorAttachs.begin(); 
         it != m_rbColorAttachs.end(); ++it)
    {
        const ColorAttachInfo &attachInfo(*it);
        attachRenderbufferImpl(attachInfo.attachment, attachInfo.format, 
                               attachInfo.oglName);
    }
    
    m_depthbuffer.storage(GL_DEPTH_COMPONENT, m_width, m_height);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);        

}

void Framebuffer::enableDepth(bool enable)
{
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_fbo);
    if ( enable )    
        glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, 
                                     GL_RENDERBUFFER_EXT, m_depthbuffer.oglName());
    else    
        glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, 0);      
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}
   
void Framebuffer::sizeFromViewport()
{
    int viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    size(viewport[2], viewport[3]);
}

void Framebuffer::begin(GLenum attachment)
{
    glPushAttrib(GL_VIEWPORT_BIT);
    glViewport(0, 0, m_width, m_height);    
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_fbo);
    glDrawBuffer(attachment);
}

void Framebuffer::begin()
{
    glPushAttrib(GL_VIEWPORT_BIT | GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, m_width, m_height);    
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_fbo);
    
    GLenum buffers[20];
    int i;
    
    for (i=0; i<static_cast<int>(m_texColorAttachs.size()) && i<20; i++)
        buffers[i] = m_texColorAttachs[i].attachment;

    for (int j=0; j<static_cast<int>(m_rbColorAttachs.size()) && i<20; j++,i++)
        buffers[i] = m_rbColorAttachs[j].attachment;

    glDrawBuffers(m_texColorAttachs.size(), buffers); 

    GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
    if ( status != GL_FRAMEBUFFER_COMPLETE_EXT )
        udDebug("Framebuffer not complete\n");
}

void Framebuffer::beginDepthmap()
{
    glPushAttrib(GL_VIEWPORT_BIT);
    glViewport(0, 0, m_width, m_height);    
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_fbo);
    glDrawBuffer(GL_FALSE);
    glReadBuffer (GL_FALSE);
}

void Framebuffer::end()
{
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    glPopAttrib();    
}

UD_NAMESPACE_END
