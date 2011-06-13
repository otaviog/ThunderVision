#include <tdvbasic/log.hpp>
#include <highgui.h>
#include "tmpbufferimage.hpp"
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
    
    void convert8UC3To32FC1Gray(const CvArr *src, CvArr *dst, CvArr *tmpRGBF)
    {
        bool inTmpRGBF = tmpRGBF == NULL;
        if ( inTmpRGBF )
        {
            CvSize sz(cvGetSize(src));
            tmpRGBF = cvCreateMat(sz.height, sz.width, CV_32FC3);
        }        
        
        cvConvertScale(src, tmpRGBF, 1.0/255.0);        
        cvCvtColor(tmpRGBF, dst, CV_RGB2GRAY);
        
        if ( inTmpRGBF )
            cvRelease(&tmpRGBF);
    }

    void convert8UC3To32FC1GrayHSV(const CvArr *src, CvArr *dst, CvArr *hsvTmp,
                                   CvArr *u8Tmp)
    {        
        cvCvtColor(src, hsvTmp, CV_RGB2HSV);
        cvSplit(hsvTmp, NULL, NULL, u8Tmp, NULL);        
                
        cvConvertScale(u8Tmp, dst, 1.0/255.0);            
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
