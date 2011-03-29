#ifndef TDV_WTADEV_HPP
#define TDV_WTADEV_HPP

#include <tdvbasic/common.hpp>
#include "optimizer.hpp"

TDV_NAMESPACE_BEGIN

class WTADev: public Optimizer
{
public:
    WTADev()
    {
    }
    
    virtual ~WTADev()
    {
    }

    void input(ReadPipe<DSIMem> *rpipe)
    {
        m_rpipe = rpipe;
    }

    ReadPipe<FloatImage>* output()
    {
        return &m_wpipe;
    }
    
    bool update();
    
private:
    ReadPipe<DSIMem> *m_rpipe;
    ReadWritePipe<FloatImage, FloatImage> m_wpipe;
};

TDV_NAMESPACE_END

#endif /* TDV_WTADEV_HPP */
