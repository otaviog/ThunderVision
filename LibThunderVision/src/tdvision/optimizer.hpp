#ifndef TDV_OPTIMIZER_HPP
#define TDV_OPTIMIZER_HPP

#include <tdvbasic/common.hpp>
#include "workunit.hpp"
#include "pipe.hpp"
#include "dsimem.hpp"
#include "floatimage.hpp"

TDV_NAMESPACE_BEGIN

class Optimizer: public WorkUnit
{
public:
    virtual void input(ReadPipe<DSIMem> *rpipe) = 0;

    virtual ReadPipe<FloatImage>* output() = 0;
};

class AbstractOptimizer: public Optimizer
{
public:
    AbstractOptimizer();
    
    void input(ReadPipe<DSIMem> *rpipe)
    {
        m_rpipe = rpipe;
    }

    ReadPipe<FloatImage>* output()
    {
        return &m_wpipe;
    }
    
    bool update();
    
protected:
    virtual void updateImpl(DSIMem mem, FloatImage img) = 0;

private:
    ReadPipe<DSIMem> *m_rpipe;
    ReadWritePipe<FloatImage, FloatImage> m_wpipe;
};

TDV_NAMESPACE_END

#endif /* TDV_OPTIMIZER_HPP */
