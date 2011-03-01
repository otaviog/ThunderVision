#ifndef TDV_TYPEDWORKUNIT_HPP
#define TDV_TYPEDWORKUNIT_HPP

#include <tdvbasic/common.hpp>
#include "workunit.hpp"
#include "pipe.hpp"

TDV_NAMESPACE_BEGIN

template<typename RT, typename WT>
class TypedWorkUnit: public WorkUnit
{
public:
    typedef RT ReadType;
    typedef WT WriteType;    
    typedef ReadPipe<RT> ReadPipeType;
    typedef WritePipe<WT> WritePipeType;
    
    TypedWorkUnit(const std::string &name)
        : WorkUnit(name)
    { }
    
    virtual ~TypedWorkUnit()
    { }
    
    void connect();
    
private:
};

TDV_NAMESPACE_END

#endif /* TDV_TYPEDWORKUNIT_HPP */
