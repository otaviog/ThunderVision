#ifndef TDV_MATCHINGCOST_HPP
#define TDV_MATCHINGCOST_HPP

#include <tdvbasic/common.hpp>
#include "workunit.hpp"
#include "pipe.hpp"
#include "dsimem.hpp"
#include "floatimage.hpp"

TDV_NAMESPACE_BEGIN

class MatchingCost: public WorkUnit
{
public:
    virtual void inputs(ReadPipe<FloatImage> *leftInput,
                        ReadPipe<FloatImage> *rightInput) = 0;
    
    virtual ReadPipe<DSIMem>* output() = 0;

};

TDV_NAMESPACE_END

#endif /* TDV_MATCHINGCOST_HPP */
