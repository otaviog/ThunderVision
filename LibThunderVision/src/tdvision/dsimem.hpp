#ifndef TDV_DSIMEM_HPP
#define TDV_DSIMEM_HPP

#include <tdvbasic/common.hpp>
#include <boost/shared_ptr.hpp>
#include "dim.hpp"

TDV_NAMESPACE_BEGIN

class DSIMemImpl
{
public:
    DSIMemImpl();    
    
    ~DSIMemImpl();
    
    void init(const Dim &dim);
    
    float* mem()
    {
        return m_mem;
    }
    
    const Dim& dim() const
    {
        return m_dim;
    }
    
private:
    Dim m_dim;
    float *m_mem;
};

class DSIMem
{
public:
    DSIMem()
    { }
    
    static DSIMem Create(const Dim &dim);

    float* mem()
    {
        return m_handle->mem();
    }
    
    const Dim& dim() const
    {
        return m_handle->dim();
    }
private:
    DSIMem(DSIMemImpl *impl)
        : m_handle(impl)
    {
    }

    boost::shared_ptr<DSIMemImpl> m_handle;     
};

TDV_NAMESPACE_END

#endif /* TDV_DSIMEM_HPP */
