#include "cameraparameters.hpp"

TDV_NAMESPACE_BEGIN

CameraParameters::CameraParameters()
{
    CvMat M = intrinsecs();
    CvMat D = distorsion();
    cvSetIdentity(&M);
    cvSetIdentity(&D);
}

TDV_NAMESPACE_END
