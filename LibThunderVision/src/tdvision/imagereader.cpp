#include <cv.h>
#include <highgui.h>
#include <tdvbasic/exception.hpp>
#include "imagereader.hpp"

TDV_NAMESPACE_BEGIN

bool ImageReader::update()
{
    WriteGuard<ReadWritePipe<CvMat*> > wg(m_wpipe);
        
    IplImage *limg = cvLoadImage(m_filename.c_str());
    
    if ( limg != NULL )
    {   
        CvMat *mat = cvCreateMat(limg->height, limg->width, CV_8UC3);
        mat = cvGetMat(limg, mat);
        
        wg.write(mat);        
    }
    else
    {
        throw Exception(boost::format("can't open image: %1%")
                        % m_filename.c_str());
    }
    
    m_wpipe.finish();
    return false;
}

void ImageReader::loadImages()
{
    if ( m_mode == Directory )
    {
        for (path::iterator it(p.begin()), it_end(p.end()); it != it_end; ++it)
            cout << "  " << *it << '\n';
    }
}

TDV_NAMESPACE_END
