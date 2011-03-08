#include "workunit.hpp"
#include "workunitprocess.hpp"

TDV_NAMESPACE_BEGIN

void WorkUnitProcess::process()
{
    while ( m_work.update() );
}

TDV_NAMESPACE_END
