#ifndef TDV_DSIMEM_HPP
#define TDV_DSIMEM_HPP

#include <tdvbasic/common.hpp>
#include <boost/shared_ptr.hpp>
#include "floatimage.hpp"
#include "dim.hpp"

TDV_NAMESPACE_BEGIN

#if 1
class DSIMemImpl
{
public:
    DSIMemImpl();    
    
    ~DSIMemImpl();
    
    void init(const Dim &dim, FloatImage lorigin);
    
    float* mem()
    {
        return m_mem;
    }
    
    const Dim& dim() const
    {
        return m_dim;
    }
    
    FloatImage leftOrigin()
    {
        return m_leftOrigin;
    }
    
private:
    Dim m_dim;
    float *m_mem;
    
    FloatImage m_leftOrigin;
};

class DSIMem
{
public:
    DSIMem()
    { }
    
    static DSIMem Create(const Dim &dim, FloatImage lorigin);

    float* mem()
    {
        if ( m_handle != NULL )
            return m_handle->mem();
        else
            return NULL;
    }
    
    const Dim& dim() const
    {
        static Dim zdim(0);
        if ( m_handle != NULL )
            return m_handle->dim();
        else
            return zdim;
    }
    
    FloatImage leftOrigin()
    {
        if ( m_handle != NULL )
            return m_handle->leftOrigin();
        else
            return FloatImage();
    }
    
private:
    DSIMem(DSIMemImpl *impl)
        : m_handle(impl)
    {
    }

    boost::shared_ptr<DSIMemImpl> m_handle;
};

#else
class DSIMem
{
public:
    DSIMem()
    { mem = NULL; }
    
    static DSIMem Create(float *mem, const Dim &dim, FloatImage lorigin);

    float* mem()
    {
        return m_handle->mem();
    }
    
    const Dim& dim() const
    {
        return m_handle->dim();
    }
    
    FloatImage leftOrigin()
    {
        return m_handle->leftOrigin();
    }
    
private:
    DSIMem()
    {
        m_mem = NULL;
    }

    float *m_mem;
};
#endif

TDV_NAMESPACE_END

#endif /* TDV_DSIMEM_HPP */
