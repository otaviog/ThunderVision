#ifndef TDV_CUDACONSTRAITS_HPP
#define TDV_CUDACONSTRAITS_HPP

#include <tdvbasic/common.hpp>
#include "dim.hpp"

TDV_NAMESPACE_BEGIN

struct WorkSize
{
    WorkSize(dim3 b, dim3 t)
        : blocks(b), threads(t)
    { }

    dim3 blocks, threads;
};

class CudaConstraits
{
public:
    WorkSize imageWorkSize(const Dim &imgDim);

private:

};

TDV_NAMESPACE_END

#endif /* TDV_CUDACONSTRAITS_HPP */
