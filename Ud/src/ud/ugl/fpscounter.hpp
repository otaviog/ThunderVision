#ifndef UD_FPSCOUNTER_HPP
#define UD_FPSCOUNTER_HPP

#include "../common.hpp"

UD_NAMESPACE_BEGIN

/**
 * Auxiliary class to count the frame rate in seconds.
 * 
 */
class FPSCounter
{
public:
    /**
     * Initializes the FPS counter. The FPS value is setted to
     * 24.0. The next call to update will set the real value.
     */
    FPSCounter();
    
    /**
     * Updates the FPS rate. If the rate is less than 1.0,
     * the FPS will continue the same.
     */
    void update();
    
    /**
     * Returns the last fps computed by update.
     */
    float get()
    {
        return static_cast<float>(m_fps);
    }

    /**
     * Returns the last fps computed by update.
     * Double version.
     */    
    double getD()
    {
        return m_fps;
    }
    
private:
    double m_fps;
    Uint32 m_lastTicks;
    int m_fcount;
};

UD_NAMESPACE_END

#endif /* UD_FPSCOUNTER_HPP */
