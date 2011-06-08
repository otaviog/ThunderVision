#include "wtadev.hpp"

TDV_NAMESPACE_BEGIN

void WTARunDev(const tdv::Dim &dsiDim, cudaPitchedPtr dsiMem, float *outimg);

void WTADev::updateImpl(DSIMem dsi, FloatImage outimg)
{            
    float *outimg_d = outimg.devMem();    
    WTARunDev(dsi.dim(), dsi.mem(), outimg_d);    
}

TDV_NAMESPACE_END
