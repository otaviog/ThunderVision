#include <cv.h>
#include <highgui.h>
#include "imagewriter.hpp"

TDV_NAMESPACE_BEGIN

bool ImageWriter::update()
{
    CvMat *fimg;     
    
    if ( m_rpipe->read(&fimg) )
    {        
        cvSaveImage(m_filename.c_str(), fimg);        
        cvDecRefData(fimg);        
        return true;
    }        
    
    return false;
}

TDV_NAMESPACE_END
