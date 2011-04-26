#include "cpustereomatcher.hpp"

TDV_NAMESPACE_BEGIN

CPUStereoMatcher::CPUStereoMatcher(StereoCorrespondenceCV::MatchingMode mode,
                     int maxDisparity, int maxIteration)
    : m_corresp(mode, maxDisparity, maxIteration)
{    
    m_correspProc.work(&m_corresp);
    m_procs.addProcess(&m_correspProc);
}

TDV_NAMESPACE_END
