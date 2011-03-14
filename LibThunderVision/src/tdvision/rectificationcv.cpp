#include <highgui.h>
#include "rectificationcv.hpp"

TDV_NAMESPACE_BEGIN

IplImage*  convert32FTo8U(IplImage *img)
{
    IplImage *convImg = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    cvConvertScale(img, convImg, 255.0);
    
    return convImg;
}

void RectificationCV::findCorners(IplImage *img, CvPoint2D32f *corners, 
                                  int *cornerCount, IplImage *eigImage, IplImage *tmpImage)
{
    static const size_t CORNER_SEARCH_WIN_DIM = 9;
        
    cvGoodFeaturesToTrack(img, eigImage, tmpImage, corners, cornerCount, 
                          0.05, 5.0);        
    
    IplImage *tmpSPImg = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    cvConvertScale(img, tmpSPImg, 255.0);
    
    cvFindCornerSubPix(tmpSPImg, corners, *cornerCount, 
                       cvSize(CORNER_SEARCH_WIN_DIM, CORNER_SEARCH_WIN_DIM),
                       cvSize(-1, -1), 
                       cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
    cvReleaseImage(&tmpSPImg);
}

bool RectificationCV::update()
{
    // http://stackoverflow.com/questions/4260594/image-rectification
    static const size_t MAX_CORNERS = 15;    
   
    WriteGuard<ReadWritePipe<FloatImage> > lwg(m_wlpipe), rwg(m_wrpipe);
    
    FloatImage limg, rimg;

    if ( m_rlpipe->read(&limg) && m_rrpipe->read(&rimg) )
    {
        IplImage *limg_c = limg.cpuMem();
        IplImage *rimg_c = rimg.cpuMem();
        
        IplImage *limg8bit_c = convert32FTo8U(limg_c);
        IplImage *rimg8bit_c = convert32FTo8U(rimg_c);
        
        const size_t maxwidth = std::max(limg.dim().width(), rimg.dim().height());
        const size_t maxheight = std::max(limg.dim().height(), rimg.dim().height());
        
        IplImage *eigImage = cvCreateImage(cvSize(maxwidth + 8, maxheight), IPL_DEPTH_32F, 1);
        IplImage *tmpImage = cvCreateImage(cvSize(maxwidth + 8, maxheight), IPL_DEPTH_32F, 1);

        CvPoint2D32f leftCorners[MAX_CORNERS], rightCorners[MAX_CORNERS];
        int leftCornerCount = MAX_CORNERS, rightCornerCount = MAX_CORNERS;
        
        findCorners(limg_c, leftCorners, &leftCornerCount,
                    eigImage, tmpImage);
        findCorners(rimg_c, rightCorners, &rightCornerCount,
                    eigImage, tmpImage);
                
        int minCornersFound = std::min(leftCornerCount, rightCornerCount);
        char status[minCornersFound];
        
        cvCalcOpticalFlowPyrLK(limg8bit_c, rimg8bit_c, eigImage, tmpImage, 
                               leftCorners, rightCorners, minCornersFound,
                               cvSize(15, 15), 1, status, NULL, 
                               cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03),
                               CV_LKFLOW_INITIAL_GUESSES);
        
        cvReleaseImage(&eigImage);
        cvReleaseImage(&tmpImage);
        
        double _fundMat[3][3];
        CvMat *leftPoints = cvCreateMat(1, minCornersFound, CV_64FC2);
        CvMat *rightPoints = cvCreateMat(1, minCornersFound, CV_64FC2);
        CvMat fundMat = cvMat(3, 3, CV_64F, _fundMat);
        
        for (int i=0; i<minCornersFound; i++)
        {
            const size_t xidx = i*2;
            const size_t yidx = xidx + 1;
            
            leftPoints->data.db[xidx] = leftCorners[i].x;
            leftPoints->data.db[yidx] = leftCorners[i].y;
            
            rightPoints->data.db[xidx] = rightCorners[i].x;
            rightPoints->data.db[yidx] = rightCorners[i].y;                        
        }
        
        const size_t minwidth = std::min(limg.dim().width(), 
                                         rimg.dim().height());
        const size_t minheight = std::min(limg.dim().height(), 
                                          rimg.dim().height());

        double _leftHomography[3][3], _rightHomography[3][3];
        double _iM[3][3], _R1[3][3], _R2[3][3], _TR[3][3];
        double _intrinsic[3][3] = { 
            {1, 0, 200}, 
            {0, 1, 100 }, 
            {0, 0, 1 } };
        
        double _distcoeffs[4] = {0, 0, 0, 0.0};
        
        CvMat intrinsic = cvMat(3, 3, CV_64F, _intrinsic);
        CvMat leftHomography = cvMat(3, 3, CV_64F, _leftHomography), 
            rightHomography = cvMat(3, 3, CV_64F, _rightHomography);
        CvMat iM = cvMat(3, 3, CV_64F, _iM);
        CvMat R1 = cvMat(3, 3, CV_64F, _R1);
        CvMat R2 = cvMat(3, 3, CV_64F, _R2);
        CvMat TR = cvMat(3, 3, CV_64F, _TR);
        CvMat distcoffs = cvMat(4, 1, CV_64F, _distcoeffs);
        
        const int fmCount = cvFindFundamentalMat(
            leftPoints, rightPoints, &fundMat);
        cvStereoRectifyUncalibrated(leftPoints, rightPoints, &fundMat, 
                                    cvSize(maxwidth, maxheight),
                                    &leftHomography,
                                    &rightHomography);        
        cvInvert(&intrinsic, &iM);
#if 0 
        cvMatMul(&leftHomography, &intrinsic, &R1);
        cvMatMul(&iM, &R1, &R1);
        
        cvMatMul(&rightHomography, &intrinsic, &R2);
        cvMatMul(&iM, &R2, &R2);
#else
        
        cvMatMul(&iM, &leftHomography, &TR);
        cvMatMul(&TR, &intrinsic, &R1);
        
        cvMatMul(&iM, &rightHomography, &TR);
        cvMatMul(&TR, &intrinsic, &R2);
#endif
        
        CvMat *mxLeft = cvCreateMat(maxheight,  maxwidth, CV_32FC1),
            *myLeft = cvCreateMat(maxheight,  maxwidth, CV_32FC1),
            *mxRight = cvCreateMat(maxheight,  maxwidth, CV_32FC1),
            *myRight = cvCreateMat(maxheight,  maxwidth, CV_32FC1);
        
        //Precompute map for cvRemap()
        cvInitUndistortRectifyMap(&intrinsic, &distcoffs, &R1, 
                                  &intrinsic, mxLeft, myLeft);  

        cvInitUndistortRectifyMap(&intrinsic, &distcoffs, &R2, 
                                  &intrinsic, mxRight, myRight);
               
        FloatImage limout = FloatImage::CreateCPU(Dim(maxwidth, maxheight));
        FloatImage rimout = FloatImage::CreateCPU(Dim(maxwidth, maxheight));
                       
        IplImage *limout_c = limout.cpuMem();
        IplImage *rimout_c = rimout.cpuMem();
        
        cvShowImage("OL", limg_c);
        cvShowImage("OR", rimg_c);
        
        cvRemap(limg_c, limout_c, mxLeft, myLeft);
        cvRemap(rimg_c, rimout_c, mxRight, myRight);
        
        cvShowImage("L", limout_c);
        cvShowImage("R", rimout_c);
        cvWaitKey(0);
        
        lwg.write(limout);
        rwg.write(rimout);
    }
    
    const bool bothWriten = lwg.wasWrite() && rwg.wasWrite();
    if ( !bothWriten )
    {
        lwg.finish();
        rwg.finish();
    }
    
    return bothWriten;
}

TDV_NAMESPACE_END
