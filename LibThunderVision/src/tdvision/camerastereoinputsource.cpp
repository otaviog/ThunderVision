#include "camerastereoinputsource.hpp"

TDV_NAMESPACE_BEGIN

CameraStereoInputSource::CameraStereoInputSource()
    : m_capture1(0),
      m_capture2(1)
{
    m_procs[0] = &m_capture1;
    m_procs[1] = &m_capture2;
    m_procs[2] = NULL;
}

TDV_NAMESPACE_END
