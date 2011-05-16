#ifndef UD_GLSTATE_HPP
#define UD_GLSTATE_HPP

#include "../common.hpp"

UD_NAMESPACE_BEGIN

template<class StateT>
class ScopedGLState
{
public:
    ScopedGLState(StateT *state)
        : m_state(state)
    {
        m_state->push();
    }

    ~ScopedGLState()
    {
        m_state->pop();
    }

private:
    StateT *m_state;
};

namespace glstate
{
    class Transparency
    {
    public:
        void push()
        {
            glPushAttrib(GL_COLOR_BUFFER_BIT);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }

        void pop()
        {
            glPopAttrib();
        }
    };
    
    class DisableLighting
    {
    public:
        void push()
        {
            glPushAttrib(GL_LIGHTING_BIT|GL_TEXTURE_BIT);
            glDisable(GL_LIGHTING);
            glShadeModel(GL_FLAT);
        }

        void pop()
        {
            glPopAttrib();
        }
    };
        
    class DisableMaterial
    {
    public:
        void push()
        {
            glPushAttrib(GL_LIGHTING_BIT|GL_TEXTURE_BIT);
            glDisable(GL_LIGHTING);
            glDisable(GL_TEXTURE_2D);
            glShadeModel(GL_FLAT);
        }

        void pop()
        {
            glPopAttrib();
        }

    };
    
    class DisableColor
    {
    public:
        void push()
        {
            glPushAttrib(GL_LIGHTING_BIT | GL_TEXTURE_BIT);
            glDisable(GL_LIGHTING);
            glDisable(GL_TEXTURE_2D);
            glShadeModel(GL_FLAT);
            glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);           
        }

        void pop()
        {
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glPopAttrib();
        }
    };
    
    class PolygonOffset
    {
    public:
        PolygonOffset(float s, float t)
            : m_s(s), m_t(t)
        { }
        
        void push()
        {
            glPushAttrib(GL_POLYGON_BIT);
            glEnable(GL_POLYGON_OFFSET_FILL);
            glPolygonOffset(m_s, m_t);
        }
        
        void pop()
        {
            glPopAttrib();
        }
    private:
        const float m_s, m_t;
    };       
}


UD_NAMESPACE_END

#endif /* UD_GLSTATE_HPP */
