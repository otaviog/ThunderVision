#ifndef TDV_DYNAMICPROGDEV_HPP
#define TDV_DYNAMICPROGDEV_HPP

#include <tdvbasic/common.hpp>
#include "workunit.hpp"
#include "pipe.hpp"

TDV_NAMESPACE_BEGIN

class DynamicProgDev: public WorkUnit
{
public:
    bool update();

private:
    
};
    
TDV_NAMESPACE_END

#endif /* TDV_DYNAMICPROGDEV_HPP */
