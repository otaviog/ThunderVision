#ifndef TDV_SSDWUDEV_HPP
#define TDV_SSDWUDEV_HPP

#include <tdvbasic/common.hpp>

TDV_NAMESPACE_BEGIN

class SSDWUDev 
{
public:    
    SSDWUDev(int disparityMax);
    
    void input(FloatImageMem left, FloatImageMem right);
    
    DSIMem output();
        
    void compute();
    
private:
};

TDV_NAMESPACE_END

#endif /* TDV_SSDWUDEV_HPP */
