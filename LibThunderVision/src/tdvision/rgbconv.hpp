#ifndef TDV_RGBCONV_HPP
#define TDV_RGBCONV_HPP

#include <tdvbasic/common.hpp>
#include <cv.h>
#include "workunit.hpp"
#include "pipe.hpp"
#include "floatimage.hpp"

TDV_NAMESPACE_BEGIN

class RGBConv: public WorkUnit
{
public:
    RGBConv();

    void input(ReadPipe<FloatImage> *rpipe)
    {
        m_rpipe = rpipe;
    }
    
    ReadPipe<IplImage*>* output()
    {
        return &m_wpipe;
    }
    
    void process();

private:
    ReadPipe<FloatImage> *m_rpipe;
    ReadWritePipe<IplImage*, IplImage*> m_wpipe;
};

TDV_NAMESPACE_END

#endif /* TDV_RGBCONV_HPP */
