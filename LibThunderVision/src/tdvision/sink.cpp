#include <tdvbasic/exception.hpp>
#include "sink.hpp"

TDV_NAMESPACE_BEGIN

void CvMatSinkPol::sink(CvMat *mat)
{   
    if ( mat != NULL )
    {
        if ( mat->refcount != NULL )
        {                            
            if ( __sync_fetch_and_sub(mat->refcount, 1) == 1 )
            {
                *mat->refcount = 1;
                cvReleaseMat(&mat);
            }
        }
        else
        {
            throw Exception("Sinking a non refcount pointer");
            //cvReleaseMat(&mat); 
        }
    }
}

void CvMatSinkPol::incrRef(CvMat *mat)
{
    cvIncRefData(mat);
}

void FloatImageSinkPol::incrRef(FloatImage img)
{
}
 
void FloatImageSinkPol::sink(FloatImage img)
{ 
    img.dispose();
}    

TDV_NAMESPACE_END
