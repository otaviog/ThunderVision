#include "thunderlang.hpp"
#include "tlctx.h"

using namespace tdv;

static CameraParameters& getCameraParms(ThunderLang &lang, const char *descId, 
    int leftOrRight)
{
    CamerasDesc &camsDesc = lang.camerasDesc(descId);    
    
    if ( leftOrRight < 1 )
        return camsDesc.leftCamera();
    else
        return camsDesc.rightCamera();    
}

static ThunderLang& getLang(void *tlc)
{    
    return reinterpret_cast<ThunderLangParser*>(tlc)->context();       
}

CEXTERN void tlcSetDistortion(void *tlc, const char *descId, int leftOrRight,
                              double d1, double d2, double d3, double d4, 
                              double d5)
{    
    CameraParameters &camParms = getCameraParms(
        getLang(tlc), descId, leftOrRight);    
    camParms.distortion(d1, d2, d3, d4, d5);
}

CEXTERN void tlcSetIntrinsic(void *tlc, const char *descId, int leftOrRight, 
                             double mtx[9])
{
   CameraParameters &camParms = getCameraParms(
        getLang(tlc), descId, leftOrRight);     
   camParms.intrinsics(mtx);   
}

CEXTERN void tlcSetExtrinsic(void *tlc, const char *descId, int leftOrRight,
                             double mtx[9])
{
   CameraParameters &camParms = getCameraParms(
        getLang(tlc), descId, leftOrRight);    
   camParms.extrinsics(mtx); 
}

CEXTERN void tlcSetFundamental(void *tlc, const char *descId, int leftOrRight,
                               double mtx[9])
{
    CamerasDesc &camsDesc = getLang(tlc).camerasDesc(descId);    
    camsDesc.fundamentalMatrix(mtx);
}
