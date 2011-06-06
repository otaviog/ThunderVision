/**
 * TODO: REDO ALL BENCHMARK CONCEPT 
 */

#ifndef TDV_BENCHMARK_HPP
#define TDV_BENCHMARK_HPP

#include <tdvbasic/common.hpp>
#include <map>
#include <vector>
#include <string>
#include <cuda_runtime.h>

TDV_NAMESPACE_BEGIN

typedef double TimeDbl;

class Benchmark
{
public:
    Benchmark()
    {
        m_secs = 0.0;
        m_timeCount = 0;
    }

    void addProbeSec(double s)
    {
        m_secs += s;
        m_timeCount += 1;
    }
    
    void addProbe(const Benchmark &bm)
    {
        addProbeSec(bm.secs());
    }
    
    double secs() const
    {
        return m_secs / double(m_timeCount);
    }
    
    Benchmark& operator += (TimeDbl tm)
    {
        addProbeSec(tm);        
        return *this;
    }
    
private:
    double m_secs;
    size_t m_timeCount;
};

class Benchmarker
{
public:
    virtual ~Benchmarker()
    { }
    
    virtual void begin() = 0;
    
    virtual void end() = 0;
    
    virtual Benchmark elapsedTime() = 0;
    
private:
};

class CudaBenchmarker: public Benchmarker
{
public:
    void begin();
    
    void end();        
    
    Benchmark elapsedTime()
    {
        return m_elapsed;
    }
    
private:
    cudaEvent_t m_evStart;
    Benchmark m_elapsed;
};
    
class BenchmarkSuite
{
public:
    BenchmarkSuite()
    { }
    
    void set(const std::string &name, const Benchmark &mark);
    
    size_t markCount() const
    {
        return m_suite.size();
    }
    
    const std::string& markName(size_t mark)
    {       
        return m_markNames[mark];
    }
    
private:
    std::map<std::string, Benchmark> m_suite;
    std::vector<std::string> m_markNames;
};

TDV_NAMESPACE_END

#endif /* TDV_BENCHMARK_HPP */
