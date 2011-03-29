#include <algorithm>
#include "dim.hpp"

TDV_NAMESPACE_BEGIN

Dim Dim::minDim(const Dim &d1, const Dim &d2)
{
    size_t dm[3] = {0, 0, 0};
    const size_t N = std::min(d1.N(), d2.N());
    for (size_t i=0; i<N; i++)
    {
        dm[i] = std::min(d1.dim[i], d2.dim[i]);
    }
    
    return Dim(dm, N);
}

TDV_NAMESPACE_END
