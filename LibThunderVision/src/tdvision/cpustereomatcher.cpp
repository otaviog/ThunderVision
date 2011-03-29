#include "cpustereomatcher.hpp"

TDV_NAMESPACE_BEGIN

CPUStereoMatcher::CPUStereoMatcher()
{    
    m_procs[0] = &m_corresp;
    m_procs[1] = NULL;
}

TDV_NAMESPACE_END
