#include "matchingcost.hpp"
#include "devstereomatcher.hpp"

TDV_NAMESPACE_BEGIN

DevStereoMatcher::DevStereoMatcher()
    : m_process(0)
{
    m_procs[0] = &m_process;
    m_procs[1] = NULL;
}

void DevStereoMatcher::inputs(ReadPipe<FloatImage> *leftInput,
                              ReadPipe<FloatImage> *rightInput)
{ 
    m_lrpipe = leftInput;
    m_rrpipe = rightInput;
    
    assert(m_matchCost != NULL);
    m_matchCost->inputs(m_lrpipe, m_rrpipe);
    
    assert(m_optimizer != NULL);
    m_optimizer->input(m_matchCost->output());
    m_cpyCPU.input(m_optimizer->output());
                   
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
