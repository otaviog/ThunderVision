#include <SDL/SDL.h>
#include "fpscounter.hpp"

UD_NAMESPACE_BEGIN

FPSCounter::FPSCounter()
{
    m_fps = 24.0;
    m_lastTicks = SDL_GetTicks();
    m_fcount = 1;
}

void FPSCounter::update()
{

    Uint32 now = SDL_GetTicks();        
    Uint32 diff = now - m_lastTicks;
    
    if ( diff > 0 )
    {
        m_fps = 1000.0/double(diff);
    }
    else
    {
        m_fps = 1000.0;
    }
    
    m_lastTicks = now;   
}

UD_NAMESPACE_END
