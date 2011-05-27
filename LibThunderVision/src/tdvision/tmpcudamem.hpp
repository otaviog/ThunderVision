#ifndef TDV_TMPCUDAMEM_HPP
#define TDV_TMPCUDAMEM_HPP

#include <tdvbasic/common.hpp>

TDV_NAMESPACE_BEGIN

class TmpCudaMem
{
public:
    TmpCudaMem();
    
    ~TmpCudaMem();
    
    float* mem(size_t size);
    
    void unalloc();
    
private:
    float *m_mem;
    size_t m_csize;
};

TDV_NAMESPACE_END

#endif /* TDV_TMPCUDAMEM_HPP */
