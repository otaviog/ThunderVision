#include <cv.h>
#include <highgui.h>
#include <tdvbasic/exception.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include "imagereader.hpp"

TDV_NAMESPACE_BEGIN

bool ImageReader::update()
{
    WriteGuard<ReadWritePipe<CvMat*> > wg(m_wpipe);
    
    if ( m_cImg < m_filenames.size() )
    {
        const std::string &filename(m_filenames[m_cImg++]);
        IplImage *img = cvLoadImage(filename.c_str());
    
        if ( img != NULL )
        {   
#if 0
            CvMat *mat = cvCreateMatHeader(img->height, img->width, CV_8UC3);
            mat = cvGetMat(img, mat);        
#else
            CvMat *mat = cvCreateMat(img->height, img->width, CV_8UC3);
            cvConvertImage(img, mat, CV_CVTIMG_SWAP_RB);
            cvReleaseImage(&img);
#endif
            wg.write(mat);
        }
        else
        {
            throw Exception(boost::format("can't open image: %1%")
                            % filename);
        }                    
    }

    return wg.wasWrite();
}

void ImageReader::loadImages()
{
    namespace fs = boost::filesystem;
        
    fs::path inpath(m_infilename);
    
    if ( fs::exists(inpath) && fs::is_directory(inpath) )
    {        
        for (fs::directory_iterator it(inpath); it != fs::directory_iterator();
             it++)
        {
            fs::path p(*it);
            if ( fs::exists(p) && fs::is_regular_file(p) )
            {
                m_filenames.push_back(p.string());
            }
        }
        
        std::sort(m_filenames.begin(), m_filenames.end());
    }
    else
    {
        m_filenames.push_back(m_infilename);
    }
}

TDV_NAMESPACE_END
