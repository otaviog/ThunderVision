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

TDV_NAMESPACE_END

#endif /* TDV_OPTIMIZER_HPP */
