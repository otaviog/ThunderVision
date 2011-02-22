#include "cuerr.hpp"
#include "cudaconstraits.hpp"

TDV_NAMESPACE_BEGIN

WorkSize CudaConstraits::imageWorkSize(const Dim &imgDim)
{
    dim3 threads(16, 16);
    dim3 blocks((imgDim.width() + 15)/16, (imgDim.height() + 15)/16);

    return WorkSize(blocks, threads);
}

TDV_NAMESPACE_END
