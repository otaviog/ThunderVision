#include "cuerr.hpp"
#include "dsimem.hpp"

TDV_NAMESPACE_BEGIN

DSIMemImpl::DSIMemImpl()
    : m_dim(-1)
{
    m_mem = NULL;
}

DSIMemImpl::~DSIMemImpl()
{
    cudaFree(m_mem);
}

void DSIMemImpl::init(const Dim &dim)
{
    CUerrExp cuerr;
    m_dim = dim;
    
    cuerr << cudaMalloc((void**) &m_mem, dim.size()*sizeof(float));    
}

DSIMem DSIMem::Create(const Dim &dim)
{
    DSIMemImpl *impl = new DSIMemImpl;
        
    try 
    {
        impl->init(dim);
        return DSIMem(impl);
    }
    catch (const std::exception &ex)
    {
        delete impl;
        throw ex;
    }                
}

TDV_NAMESPACE_END
