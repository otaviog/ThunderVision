#include "wtadev.hpp"

void DevWTARun(float *dsi, const tdv::Dim &dim, 
               float *outimg);

TDV_NAMESPACE_BEGIN

void WTADev::updateImpl(DSIMem dsi, FloatImage outimg)
{            
    float *outimg_d = outimg.devMem();    
    DevWTARun(dsi.mem(), dsi.dim(), outimg_d);    
}

TDV_NAMESPACE_END
