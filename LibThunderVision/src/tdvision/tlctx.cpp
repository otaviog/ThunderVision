#include "thunderlang.hpp"
#include "tlctx.h"

using namespace tdv;

static CameraParameters& getCameraParms(ThunderSpec &lang, const char *descId, 
    int leftOrRight)
{
    CamerasDesc &camsDesc = lang.camerasDesc(descId);    
    
    if ( leftOrRight < 1 )
        return camsDesc.leftCamera();
    else
        return camsDesc.rightCamera();    
}

static ThunderSpec& getLang(void *tlc)
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

CEXTERN void tlcSetExtrinsic(void *tlc, const char *descId, double rot[9], 
                             double t1, double t2, double t3)
{
    CamerasDesc &camsDesc = getLang(tlc).camerasDesc(descId);    
    const double loc[3] = {t1, t2, t3};
    camsDesc.extrinsics(rot, loc);

}

CEXTERN void tlcSetFundamental(void *tlc, const char *descId, double mtx[9])
{
    CamerasDesc &camsDesc = getLang(tlc).camerasDesc(descId);    
    camsDesc.fundamentalMatrix(mtx);
}
