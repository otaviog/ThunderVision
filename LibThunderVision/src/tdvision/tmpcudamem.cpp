#include "cuerr.hpp"
#include "tmpcudamem.hpp"

TDV_NAMESPACE_BEGIN

TmpCudaMem::TmpCudaMem()
{    
    m_mem_d = NULL;
    m_bsize = 0;
}
    
TmpCudaMem::~TmpCudaMem()
{
    unalloc();
}

void* TmpCudaMem::mem(size_t bsize)
{    
    if ( m_bsize != bsize )
    {
        unalloc();
        
        CUerrExp cuerr;
        
        cuerr << cudaMalloc((void**) &m_mem_d, bsize);
        m_bsize = bsize;
    }    
    
    return m_mem_d;
}

void TmpCudaMem::unalloc()
{
    if ( m_mem_d != NULL )
    {
        cudaFree(m_mem_d);
        m_mem_d = NULL;
    }
}

TDV_NAMESPACE_END
