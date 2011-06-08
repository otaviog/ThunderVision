#ifndef TDV_TMPCUDAMEM_HPP
#define TDV_TMPCUDAMEM_HPP

#include <tdvbasic/common.hpp>
#include <cuda_runtime.h>
#include "dim.hpp"

TDV_NAMESPACE_BEGIN

class TmpCudaMem
{
public:
    TmpCudaMem();
    
    ~TmpCudaMem();
    
    void* mem(size_t bsize);
    
    void unalloc();
    
private:
    void* m_mem_d;
    size_t m_bsize;
};

TDV_NAMESPACE_END

#endif /* TDV_TMPCUDAMEM_HPP */
