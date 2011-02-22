#ifndef TDV_WORKUNIT_HPP
#define TDV_WORKUNIT_HPP

#include <tdvbasic/common.hpp>
#include <string>

TDV_NAMESPACE_BEGIN

class WorkUnit
{
public:
    WorkUnit(const std::string &nm)        
        : m_name(nm)
    {
    }
    
    virtual ~WorkUnit()
    { }
        
    virtual void process() = 0;
        
    const std::string& name() const
    {
        return m_name;
    }
    
private:    
    std::string m_name;
};

TDV_NAMESPACE_END

#endif /* TDV_WORKUNIT_HPP */
