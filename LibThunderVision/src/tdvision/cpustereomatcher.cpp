#include "cpustereomatcher.hpp"

TDV_NAMESPACE_BEGIN

CPUStereoMatcher::CPUStereoMatcher()
{    
    m_procs.addProcess(&m_corresp);            
}

TDV_NAMESPACE_END
