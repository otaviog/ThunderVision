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
                                            dim.height(), dim.width());
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
        m_dim = Dim(0);
    }
}

void* LocalDSIMem::toCpuMem()
{    
    if ( m_mem.ptr == NULL )    
        return NULL;    
    
    unsigned char *cpuMem = new unsigned char[m_dim.size()*m_typeSize];
    const size_t pitch = m_mem.pitch;
    const size_t width = m_dim.width();    
    const size_t height = m_dim.height();
    const size_t depth = m_dim.depth();
    
    CUerrExp cuerr;
    try
    {
        for ( size_t slice=0; slice<m_dim.width(); slice++)
        {
            for (size_t row=0; row<height; row++)
            {
                const size_t devOffset = pitch*height*slice + pitch*row;
                const size_t cpuOffset = depth*height*slice + depth*row;
                assert(cpuOffset < m_dim.size()*m_typeSize);
                assert(devOffset < pitch*height*width*m_typeSize);
                
                cuerr << cudaMemcpy(cpuMem + cpuOffset*m_typeSize, 
                                    ((unsigned char*) m_mem.ptr) + devOffset, 
                                    m_mem.pitch,
                                    cudaMemcpyDeviceToHost);
            }
        }
    }
    catch ( const CUException &ex)
    {
        delete [] cpuMem;
        throw ex;
    }    
    
    return cpuMem;
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
