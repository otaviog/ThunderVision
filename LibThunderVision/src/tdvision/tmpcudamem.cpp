#include "cuerr.hpp"
#include "tmpcudamem.hpp"

TDV_NAMESPACE_BEGIN

TmpCudaMem::TmpCudaMem()
{
    m_csize = 0;
    m_mem = NULL;
}
    
TmpCudaMem::~TmpCudaMem()
{
    if ( m_mem != NULL )
    {
        cudaFree(m_mem);
        m_mem = NULL;
    }
}

float* TmpCudaMem::mem(size_t memSize)
{
    CUerrExp cuerr;
    
    if ( m_csize != memSize )
    {
        cudaFree(m_mem);
        m_mem = NULL;
    }
    
    cuerr << cudaMalloc((void**) &m_mem, memSize);
    return m_mem;
}

void TmpCudaMem::unalloc()
{
    if ( m_mem != NULL )
    {
        cudaFree(m_mem);
        m_mem = NULL;
    }
}

TDV_NAMESPACE_END
