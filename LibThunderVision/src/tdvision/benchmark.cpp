#include <cuda_runtime.h>
#include "cuerr.hpp"
#include "benchmark.hpp"

TDV_NAMESPACE_BEGIN

void CudaBenchmarker::begin()
{   
    CUerrExp err;
    
    CUerrDB(cudaEventCreate(&m_evStart));
    CUerrDB(cudaEventRecord(m_evStart, 0));
}

TimeDbl CudaBenchmarker::end()
{
    CUerrExp err;
    
    cudaEvent_t stop;
    CUerrDB(cudaEventCreate(&stop));
    CUerrDB(cudaEventRecord(stop, 0));
    CUerrDB(cudaEventSynchronize(stop));
    
    float time;
    CUerrDB(cudaEventElapsedTime(&time, m_evStart, stop));

    err<<cudaEventDestroy(m_evStart);
    CUerrDB(cudaEventDestroy(stop));

    return time;
}

void BenchmarkSuite::set(const std::string &name, const Benchmark &mark)
{
    if ( !m_suite.count(name) )
        m_markNames.push_back(name);

    m_suite[name] = mark;    
}

TDV_NAMESPACE_END
