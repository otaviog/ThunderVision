#ifndef TDV_DEVSTEREOMATCHER_HPP
#define TDV_DEVSTEREOMATCHER_HPP

#include <boost/shared_ptr.hpp>
#include <tdvbasic/common.hpp>
#include "cudaprocess.hpp"
#include "stereomatcher.hpp"
#include "optimizer.hpp"

TDV_NAMESPACE_BEGIN

class MatchingCost;

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
        assert(m_optimizer != NULL);
        return m_optimizer->output();
    }
    
    std::string name() const;

private:
    ReadPipe<FloatImage> *m_lrpipe, *m_rrpipe;
    boost::shared_ptr<MatchingCost> m_matchCost;
    boost::shared_ptr<Optimizer> m_optimizer;
    
    CUDAProcess m_process;
    Process *m_procs[2];
};

TDV_NAMESPACE_END

#endif /* TDV_DEVSTEREOMATCHER_HPP */
