#ifndef TDV_CPYTOCPU_HPP
#define TDV_CPYTOCPU_HPP

#include <tdvbasic/common.hpp>
#include "pipe.hpp"
#include "workunit.hpp"
#include "floatimage.hpp"

TDV_NAMESPACE_BEGIN

class CpyImageToCPU: public WorkUnit
{
public:
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
    ReadWritePipe<FloatImage, FloatImage> m_wpipe;
};

TDV_NAMESPACE_END

#endif /* TDV_CPYTOCPU_HPP */
