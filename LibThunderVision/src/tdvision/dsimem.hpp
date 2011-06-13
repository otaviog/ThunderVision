#ifndef TDV_DSIMEM_HPP
#define TDV_DSIMEM_HPP

#include <tdvbasic/common.hpp>
#include <boost/shared_ptr.hpp>
#include <cuda.h>
#include "floatimage.hpp"
#include "dim.hpp"

TDV_NAMESPACE_BEGIN

class LocalDSIMem
{
public:    
    LocalDSIMem(size_t typeSize = sizeof(float));
    
    ~LocalDSIMem();

    const Dim& dim() const
    {
        return m_dim;
    }
    
    cudaPitchedPtr mem()
    {
        return m_mem;
    }
    
    cudaPitchedPtr mem(const Dim &dim);
    
    void unalloc();
    
    void* toCpuMem();
    
private:
    Dim m_dim;
    cudaPitchedPtr m_mem;
    size_t m_typeSize;
};

class DSIMemImpl
{
public:
    DSIMemImpl()
    { }
    
    ~DSIMemImpl()
    { }
    
    void init(const Dim &dim, FloatImage lorigin);
    
    cudaPitchedPtr mem()
    {
        return m_dsiMem.mem();
    }
    
    void* toCpuMem()
    {
        m_dsiMem.toCpuMem();
    }
    
    const Dim& dim() const
    {
        return m_dsiMem.dim();
    }
    
    FloatImage leftOrigin()
    {
        return m_leftOrigin;
    }
    
    void leftOrigin(FloatImage img)
    {
        m_leftOrigin = img;
    }

private:
    LocalDSIMem m_dsiMem;
    FloatImage m_leftOrigin;
};

class DSIMem
{
public:
    DSIMem()
    { }
    
    static DSIMem Create(const Dim &dim, FloatImage lorigin);

    cudaPitchedPtr mem()
    {
        if ( m_handle != NULL )
            return m_handle->mem();
        else
            return cudaPitchedPtr();
    }
    
    void* toCpuMem() 
    {
        if ( m_handle != NULL )
            return m_handle->toCpuMem();
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

    void leftOrigin(FloatImage img)
    {
        if ( m_handle != NULL )
            m_handle->leftOrigin(img);
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
