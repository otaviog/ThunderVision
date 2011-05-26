#include "matchingcost.hpp"
#include "devstereomatcher.hpp"

TDV_NAMESPACE_BEGIN

DevStereoMatcher::DevStereoMatcher()
    : m_process(0)
{
    m_procs[0] = &m_process;
    m_procs[1] = NULL;
    m_useMedianfilter = false;
}

void DevStereoMatcher::inputs(ReadPipe<FloatImage> *leftInput,
                              ReadPipe<FloatImage> *rightInput)
{ 
    m_lrpipe = leftInput;
    m_rrpipe = rightInput;
    
    assert(m_matchCost != NULL);
    
    if ( m_useMedianfilter )
    {
        m_medianFilter[0].input(m_lrpipe);
        m_medianFilter[1].input(m_rrpipe);
        
        m_matchCost->inputs(m_medianFilter[0].output(), 
                            m_medianFilter[1].output());
    }
    else
    {
        m_matchCost->inputs(m_lrpipe, m_rrpipe);
    }
    
    
    assert(m_optimizer != NULL);
    m_optimizer->input(m_matchCost->output());
    m_cpyCPU.input(m_optimizer->output());

    if ( m_useMedianfilter )
    {
        m_process.addWork(&m_medianFilter[0]);
        m_process.addWork(&m_medianFilter[1]);
    }
    
    m_process.addWork(m_matchCost.get());
    m_process.addWork(m_optimizer.get()); 
    m_process.addWork(&m_cpyCPU);
}

std::string DevStereoMatcher::name() const
{    
    return ((m_matchCost != NULL) 
            ? m_matchCost->workName() : std::string(""))
        + std::string("+") 
        + ((m_optimizer != NULL) ? m_optimizer->workName() : std::string(""));
}

TDV_NAMESPACE_END
