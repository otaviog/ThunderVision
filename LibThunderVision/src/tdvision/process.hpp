
#ifndef TDV_PROCESS_HPP
#define TDV_PROCESS_HPP

#include <tdvbasic/common.hpp>

TDV_NAMESPACE_BEGIN

class Process
{
public:
    virtual void process() = 0;    
    
    virtual void finish()
    { }

private:
};

TDV_NAMESPACE_END

#endif /* TDV_PROCESS_HPP */

