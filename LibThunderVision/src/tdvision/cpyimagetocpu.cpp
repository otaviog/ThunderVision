#include "cpyimagetocpu.hpp"

TDV_NAMESPACE_BEGIN

FloatImage CpyImageToCPU::updateImpl(FloatImage input)
{
    input.disposeFromDev();
    input.cpuMem();
    return input;
}

TDV_NAMESPACE_END
