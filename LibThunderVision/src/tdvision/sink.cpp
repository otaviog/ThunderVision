#include "sink.hpp"

TDV_NAMESPACE_BEGIN

void CvMatSinkPol::sink(CvMat *mat)
{        
    cvDecRefData(mat);
        
    if ( mat->refcount <= 0 )
    {
        cvReleaseMat(&mat);
    }
}

void CvMatSinkPol::incrRef(CvMat *mat)
{
    cvIncRefData(mat);
}

TDV_NAMESPACE_END
