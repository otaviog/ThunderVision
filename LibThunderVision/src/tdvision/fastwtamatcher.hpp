#ifndef TDV_FASTWTAMATCHER_HPP
#define TDV_FASTWTAMATCHER_HPP

#include <tdvbasic/common.hpp>
#include "stereomatcher.hpp"
#include "process.hpp"

TDV_NAMESPACE_BEGIN

class FastWTAMatcher: public StereoMatcher, public Process
{
public:
    FastWTAMatcher(int maxDisparity);
    
    void inputs(ReadPipe<FloatImage> *leftInput,
                ReadPipe<FloatImage> *rightInput);
    
    Process** processes()
    {
        return m_procs;
    }        
    
    Benchmark matchcostBenchmark() const
    {
        return m_mark;
    }
    
    Benchmark optimizationBenchmark() const
    {
        return m_mark;
    }

    ReadPipe<FloatImage>* output()
    {        
        return &m_wpipe;
    }
    
    std::string name() const
    {
        return "FastWTAMatcher";
    }
    
    void process();
    
    bool update();
    
private:
    ReadPipe<FloatImage> *m_lrpipe, *m_rrpipe;
    ReadWritePipe<FloatImage> m_wpipe;
    Benchmark m_mark;

    Process *m_procs[2];
    
    int m_maxDisparity;
};

TDV_NAMESPACE_END

#endif /* TDV_FASTWTAMATCHER_HPP */
