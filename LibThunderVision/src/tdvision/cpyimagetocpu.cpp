#include "cpyimagetocpu.hpp"

TDV_NAMESPACE_BEGIN

bool CpyImageToCPU::update()
{
    WriteGuard<WritePipe<FloatImage> > guard(m_wpipe);
    
    FloatImage input;
    if ( m_rpipe->read(&input) )
    {
        IplImage *img = input.cpuMem();
        guard.write(input);
    }
    
    return guard.wasWrite();
}

TDV_NAMESPACE_END
