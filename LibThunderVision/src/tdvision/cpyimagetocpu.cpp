#include "cpyimagetocpu.hpp"

TDV_NAMESPACE_BEGIN

FloatImage CpyImageToCPU::updateImpl(FloatImage input)
{    
    input.cpuMem();
    input.disposeFromDev();
    return input;
}

TDV_NAMESPACE_END
