#include "dsimem.hpp"
#include "cuerr.hpp"

#define DSIMEM_USE_ALIGNED 

TDV_NAMESPACE_BEGIN

LocalDSIMem::LocalDSIMem(size_t typeSize)
    : m_dim(0)
{
    m_mem.ptr = NULL;
    m_typeSize = typeSize;
}

LocalDSIMem::~LocalDSIMem()
{
    unalloc();
}

cudaPitchedPtr LocalDSIMem::mem(const Dim &dim)
{
    CUerrExp cuerr;
    if ( m_dim != dim )
    {
        unalloc();

#ifdef DSIMEM_USE_ALIGNED
        cudaExtent extent = make_cudaExtent(dim.depth()*m_typeSize,
                                            dim.width(), dim.height());
        cuerr << cudaMalloc3D(&m_mem, extent);
#else
        cuerr << cudaMalloc(&m_mem.ptr, dim.size()*m_typeSize);
        m_mem.pitch = dim.depth()*m_typeSize;
#endif

        m_dim = dim;
    }

    return m_mem;
}

void LocalDSIMem::unalloc()
{
    CUerrExp cuerr;
    if ( m_mem.ptr != NULL )
    {
        cuerr << cudaFree(m_mem.ptr);
        m_mem.ptr = NULL;
    }
}

void DSIMemImpl::init(const Dim &dim, FloatImage lorigin)
{
    m_dsiMem.mem(dim);
    m_leftOrigin = lorigin;
}

DSIMem DSIMem::Create(const Dim &dim, FloatImage lorigin)
{
    DSIMemImpl *impl = new DSIMemImpl;

    try
    {
        impl->init(dim, lorigin);
        return DSIMem(impl);
    }
    catch (const std::exception &ex)
    {
        delete impl;
        throw ex;
    }
}

TDV_NAMESPACE_END
