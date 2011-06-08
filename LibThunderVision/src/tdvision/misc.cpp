#include <tdvbasic/log.hpp>
#include <highgui.h>
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
#if 0
        
        CvMat *grayTmp = create8UGray(src);
        cvConvertScale(grayTmp, dst, 1.0/255.0);
        cvReleaseMat(&grayTmp);       
#elif 1
            
        CvMat *tmp = cvCreateMat(sz.height, sz.width, CV_32FC3);
        
        cvConvertScale(src, tmp, 1.0/255.0);        
        cvCvtColor(tmp, dst, CV_RGB2GRAY);
        cvReleaseMat(&tmp);
        
#else
        CvMat *hsv = cvCreateMat(sz.height, sz.width, CV_8UC3);
        CvMat *tmp = cvCreateMat(sz.height, sz.width, CV_8U);
        
        cvCvtColor(src, hsv, CV_RGB2HSV);
        cvSplit(hsv, NULL, NULL, tmp, NULL);        
                
        cvConvertScale(tmp, dst, 1.0/255.0);
            
        cvReleaseMat(&hsv);
        cvReleaseMat(&tmp);
#endif

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

#if 0
void showDiagImg(int w, int h, SGPath *path, size_t pathCount)
{
    int scale = 8;
    CvMat *mat = cvCreateMat(h*scale, w*scale, CV_8UC3);
    cvRectangle(mat, 
                cvPoint(0, 0), 
                cvPoint((w - 1)*scale, (h - 1)*scale), 
                CV_RGB(0, 0, 0), -1);
          
    for (int i=0; i < pathCount; i++) 
    {
        cvDrawLine(mat, 
                   cvPoint(path[i].start.x*scale, 
                           path[i].start.y*scale),
                   cvPoint(path[i].end.x*scale, 
                           path[i].end.y*scale), 
                   CV_RGB(255, 0, 0));
    }
    cvSaveImage("diags.png", mat);
    cvReleaseMat(&mat);
}
#endif
TDV_NAMESPACE_END
