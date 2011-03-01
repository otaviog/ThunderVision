#ifndef TDV_RESIZEIMAGE_HPP
#define TDV_RESIZEIMAGE_HPP

#include <tdvbasic/common.hpp>
#include "typedworkunit.hpp"
#include "pipe.hpp"
#include "floatimage.hpp"

TDV_NAMESPACE_BEGIN

class ResizeImageWU: public TypedWorkUnit<FloatImage, FloatImage>
{
    enum Mode
    {
        Absolute, Percent, NextPowerOf2
    };

public:    

    ResizeImageWU()
        : TypedWorkUnit<FloatImage, FloatImage>("Resize Image"),
          m_newDim(-1)
    {        
        m_mode = NextPowerOf2;
        m_rpipe = NULL; 
        m_wpipe = NULL;
    }
        
    ResizeImageWU(const Dim &newDim)
        : TypedWorkUnit<FloatImage, FloatImage>("Resize Image"),
          m_newDim(newDim)
    {
        m_mode = Absolute;
        m_rpipe = NULL; 
        m_wpipe = NULL;
    }
    
    ResizeImageWU(float xpercent, float ypercent)
        : TypedWorkUnit<FloatImage, FloatImage>("Resize Image"),
          m_newDim(-1)
    {
        m_xpercent = xpercent;
        m_ypercent = ypercent;
        
        m_mode = Percent;
        
        m_rpipe = NULL; 
        m_wpipe = NULL;
    }
    
    void input(ReadPipe<FloatImage> *rpipe)
    {
        m_rpipe = rpipe;
    }

    void output(WritePipe<FloatImage> *wpipe)
    {
        m_wpipe = wpipe;
    }
    
    void process();
    
private:
    ReadPipe<FloatImage> *m_rpipe;
    WritePipe<FloatImage> *m_wpipe;
    
    Mode m_mode;
    const Dim &m_newDim;
    float m_xpercent, m_ypercent;    
};

TDV_NAMESPACE_END

#endif /* TDV_RESIZEIMAGE_HPP */

