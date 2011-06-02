#ifndef TDV_DEVSTEREOMATCHER_HPP
#define TDV_DEVSTEREOMATCHER_HPP

#include <boost/shared_ptr.hpp>
#include <tdvbasic/common.hpp>
#include "cudaprocess.hpp"
#include "stereomatcher.hpp"
#include "matchingcost.hpp"
#include "optimizer.hpp"
#include "cpyimagetocpu.hpp"
#include "medianfilterdev.hpp"
#include "medianfiltercpu.hpp"

TDV_NAMESPACE_BEGIN

class DevStereoMatcher: public StereoMatcher
{
public:
    DevStereoMatcher();
        
    void inputs(ReadPipe<FloatImage> *leftInput,
                ReadPipe<FloatImage> *rightInput);

    Process** processes()
    {
        return m_procs;
    }
    
    void setMatchingCost(boost::shared_ptr<MatchingCost> matchCost)
    {
        m_matchCost = matchCost;
    }

    void setOptimizer(boost::shared_ptr<Optimizer> optimizer)
    {
        m_optimizer = optimizer;
    }
    
    ReadPipe<FloatImage>* output()
    {        
        return m_cpyCPU.output();
    }
        
    std::string name() const;
    
    Benchmark matchcostBenchmark() const    
    {        
        return (m_matchCost != NULL)
            ? m_matchCost->benchmark()
            : Benchmark();                    
    }
    
    Benchmark optimizationBenchmark() const
    {
        return (m_optimizer != NULL)
            ? m_optimizer->benchmark()
            : Benchmark();
    }

private:
    ReadPipe<FloatImage> *m_lrpipe, *m_rrpipe;
    // TODO: Use device median filter.
    //MedianFilterDev m_medianFilter[2];    
    MedianFilterCPU m_medianFilter[2];
    boost::shared_ptr<MatchingCost> m_matchCost;
    boost::shared_ptr<Optimizer> m_optimizer;
    CpyImageToCPU m_cpyCPU;
    CUDAProcess m_process;    
    Process *m_procs[2];
    
    bool m_useMedianfilter;
};

TDV_NAMESPACE_END

#endif /* TDV_DEVSTEREOMATCHER_HPP */
