#include "misc.hpp"

TDV_NAMESPACE_BEGIN

namespace misc
{
    CvMat* create8UGray(const CvArr *src)
    {
        CvSize sz(cvGetSize(src));
        CvMat *dst = cvCreateMat(sz.height, sz.width, CV_8UC1);
        cvCvtColor(src, dst, CV_RGB2GRAY);
        return dst;
    }
    
    CvMat *create32FGray(const CvArr *src)
    {                        
        CvSize sz(cvGetSize(src));
        CvMat *dst = cvCreateMat(sz.height, sz.width, CV_32F);
        
        CvMat *grayTmp = create8UGray(src);
        cvConvertScale(grayTmp, dst, 1.0/255.0);
        cvReleaseMat(&grayTmp);
        
        return dst;
    }

    void convert8UC3To32FC1Gray(const CvArr *src, CvArr *dst, CvArr *tmpGray)
    {
        bool inTmpGrayNull = tmpGray == NULL;
        if ( inTmpGrayNull )
        {
            tmpGray = misc::create8UGray(src);
        }
        else
        {
            cvCvtColor(src, tmpGray, CV_RGB2GRAY);
        }
        
        cvConvertScale(tmpGray, dst, 1.0/255.0);
        
        if ( inTmpGrayNull )
            cvReleaseMat((CvMat**) &tmpGray);
    }
}

TDV_NAMESPACE_END
