#ifndef TDV_SEMIGLOBALLABMATCHER_HPP
#define TDV_SEMIGLOBALLABMATCHER_HPP

#include <tdvbasic/common.hpp>
#include "stereomatcher.hpp"
#include "process.hpp"
#include "semiglobal.h"
#include "dsimem.hpp"

TDV_NAMESPACE_BEGIN

class SemiglobalLabMatcher: public StereoMatcher, public Process
{
public:
    SemiglobalLabMatcher(int maxDisparity);
    
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
        return "SemiglobalLabMatcher";
    }
    
    void process();
    
    bool update();
    
private:
    ReadPipe<FloatImage> *m_lrpipe, *m_rrpipe;
    ReadWritePipe<FloatImage> m_wpipe;
    Benchmark m_mark;

    Process *m_procs[2];
    
    int m_maxDisparity;

    LocalDSIMem m_aggregDSI;
    bool m_zeroAggregDSI;    
    SGPaths m_sgPaths;

};

TDV_NAMESPACE_END

#endif /* TDV_SEMIGLOBALLABMATCHER_HPP */
