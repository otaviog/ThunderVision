#ifndef TDV_IMAGERESIZE_HPP
#define TDV_IMAGERESIZE_HPP

#include <tdvbasic/common.hpp>
#include "workunit.hpp"
#include "pipe.hpp"
#include "floatimage.hpp"

TDV_NAMESPACE_BEGIN

class ImageResize: public WorkUnit
{
    enum Mode
    {
        Absolute, 
        Percent, 
        NextPowerOf2, 
        ClosestPowerOf2
    };

public:    

    ImageResize()
        : m_newDim(-1)
    {     
        workName("Image resize");
        m_mode = ClosestPowerOf2;
        m_rpipe = NULL;         
    }
        
    ImageResize(const Dim &newDim)
        : m_newDim(newDim)
    {
        workName("Image resize");
        m_mode = Absolute;
        m_rpipe = NULL; 
    }
    
    ImageResize(float xpercent, float ypercent)
        : m_newDim(-1)
    {
        workName("Image resize");
        
        m_xpercent = xpercent;
        m_ypercent = ypercent;
        
        m_mode = Percent;
        
        m_rpipe = NULL; 
    }
    
    void input(ReadPipe<FloatImage> *rpipe)
    {
        m_rpipe = rpipe;
    }

    ReadPipe<FloatImage>* output()
    {
        return &m_wpipe;
    }
    
    bool update();
    
private:
    ReadPipe<FloatImage> *m_rpipe;
    ReadWritePipe<FloatImage> m_wpipe;
    
    Mode m_mode;
    const Dim &m_newDim;
    float m_xpercent, m_ypercent;    
};

TDV_NAMESPACE_END

#endif /* TDV_IMAGERESIZE_HPP */

