#include "processgroup.hpp"

TDV_NAMESPACE_BEGIN

void ArrayProcessGroup::addProcess(Process **procs)
{
    size_t procCount = 0;
    
    while ( procs[procCount] != NULL )
    {        
        addProcess(procs[procCount]);
        procCount++;
    }
}

TDV_NAMESPACE_END
