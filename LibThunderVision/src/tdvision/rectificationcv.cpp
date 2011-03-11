#include "rectificationcv.hpp"

TDV_NAMESPACE_BEGIN

void RectificationCV::findCorners(IplImage *img, CvPoint2D32f *corners, 
                                  int *cornerCount, IplImage *eigImage, IplImage *tmpImage)
{
    static const size_t CORNER_SEARCH_WIN_DIM = 9;
    cvGoodFeaturesToTrack(img, eigImage, tmpImage, corners, cornerCount, 
                          0.05, 5.0);        
    cvFindCornerSubPix(img, corners, *cornerCount, 
                       cvSize(CORNER_SEARCH_WIN_DIM, CORNER_SEARCH_WIN_DIM),
                       cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
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
        
        cvCalcOpticalFlowPyrLK(limg_c, rimg_c, eigImage, tmpImage, 
                               leftCorners, rightCorners, minCornersFound,
                               cvSize(15, 15), 1, status, NULL, 
                               cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03),
                               CV_LKFLOW_INITIAL_GUESSES);
        
        cvReleaseImage(&eigImage);
        cvReleaseImage(&tmpImage);

        CvMat *leftPoints = cvCreateMat(1, minCornersFound, CV_32FC2);
        CvMat *rightPoints = cvCreateMat(1, minCornersFound, CV_32FC2);        
        CvMat *fundStatus = cvCreateMat(1, minCornersFound, CV_8UC1);        
        CvMat *fundMat = cvCreateMat(3, 3, CV_32FC1);
        
        for (int i=0; i<minCornersFound; i++)
        {
            const size_t xidx = i*2;
            const size_t yidx = xidx + 1;
            
            leftPoints->data.db[xidx] = leftCorners[i].x;
            leftPoints->data.db[yidx] = leftCorners[i].y;
            
            rightPoints->data.db[xidx] = rightCorners[i].x;
            rightPoints->data.db[yidx] = rightCorners[i].y;                        
        }
        
        const size_t minwidth = std::min(limg.dim().width(), rimg.dim().height());
        const size_t minheight = std::min(limg.dim().height(), rimg.dim().height());

        CvMat *leftHomography = cvCreateMat(3, 3, CV_32FC1), 
            *rightHomography = cvCreateMat(3, 3, CV_32FC1);
        
        cvFindFundamentalMat(leftPoints, rightPoints, fundMat);
        cvStereoRectifyUncalibrated(leftPoints, rightPoints, fundMat, 
                                    cvSize(maxwidth, maxheight),
                                    leftHomography,
                                    rightHomography);
                                    
                                    
        
    }
    
}

TDV_NAMESPACE_END
