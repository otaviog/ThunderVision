#include <highgui.h>
#include <tdvbasic/log.hpp>
#include "rectificationcv.hpp"

TDV_NAMESPACE_BEGIN

static IplImage* convert32FTo8U(const IplImage *img)
{
    IplImage *convImg = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    cvConvertScale(img, convImg, 255.0);

    return convImg;
}

void RectificationCV::findCorners(const IplImage *img, CvPoint2D32f *corners,
                                  int *cornerCount, IplImage *eigImage,
                                  IplImage *tmpImage)
{
    static const size_t CORNER_SEARCH_WIN_DIM = 9;

    cvGoodFeaturesToTrack(img, eigImage, tmpImage, corners, cornerCount,
                          0.05, 5.0);

    IplImage *tmpSPImg = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    cvConvertScale(img, tmpSPImg, 255.0);

    cvFindCornerSubPix(tmpSPImg, corners, *cornerCount,
                       cvSize(CORNER_SEARCH_WIN_DIM, CORNER_SEARCH_WIN_DIM),
                       cvSize(-1, -1),
                       cvTermCriteria(
                           CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
    cvReleaseImage(&tmpSPImg);
}

size_t RectificationCV::findCornersPoints(const IplImage *limg_c, const IplImage *rimg_c,
                                          const CvSize &imgSize, CvMat **leftPointsR,
                                          CvMat **rightPointsR)
{
    static const size_t MAX_CORNERS = 15;

    IplImage *limg8bit_c = convert32FTo8U(limg_c);
    IplImage *rimg8bit_c = convert32FTo8U(rimg_c);

    IplImage *eigImage = cvCreateImage(cvSize(imgSize.width + 8, imgSize.height), IPL_DEPTH_32F, 1);
    IplImage *tmpImage = cvCreateImage(cvSize(imgSize.width + 8, imgSize.height), IPL_DEPTH_32F, 1);

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

    CvMat *leftPoints = cvCreateMat(1, minCornersFound, CV_64FC2);
    CvMat *rightPoints = cvCreateMat(1, minCornersFound, CV_64FC2);

    for (int i=0; i<minCornersFound; i++)
    {
        const size_t xidx = i*2;
        const size_t yidx = xidx + 1;

        leftPoints->data.db[xidx] = leftCorners[i].x;
        leftPoints->data.db[yidx] = leftCorners[i].y;

        rightPoints->data.db[xidx] = rightCorners[i].x;
        rightPoints->data.db[yidx] = rightCorners[i].y;
    }

    *leftPointsR = leftPoints;
    *rightPointsR = rightPoints;

    return minCornersFound;
}

void RectificationCV::calibratedRectify(const CvMat *lM, const CvMat *rM,
                                        const CvMat *lD, const CvMat *rD,
                                        const CvMat *R, const CvMat *T,
                                        const CvMat *F, const CvSize &imgSize,
                                        CvMat *mxLeft, CvMat *myLeft,
                                        CvMat *mxRight, CvMat *myRight)
{
    double c_lR[9], c_rR[9], c_lP[12], c_rP[12];
    CvMat v_lR = cvMat(3, 3, CV_64F, c_lR);
    CvMat v_rR = cvMat(3, 3, CV_64F, c_rR);
    CvMat v_lP = cvMat(3, 4, CV_64F, c_lP);
    CvMat v_rP = cvMat(3, 4, CV_64F, c_rP);

    cvSetIdentity(&v_lR);
    cvSetIdentity(&v_rR);
    cvSetIdentity(&v_lP);
    cvSetIdentity(&v_rP);

    cvStereoRectify(lM, rM, lD, rD, imgSize, R, T,
                    &v_lR, &v_rR, &v_lP, &v_rP);

    cvInitUndistortRectifyMap(lM, lD, &v_lR, &v_lP,
                              mxLeft, myLeft);
    cvInitUndistortRectifyMap(rM, rD, &v_rR, &v_rP,
                              mxRight, myRight);
}

void RectificationCV::uncalibrateRectify(
    const CvMat *leftPoints, const CvMat *rightPoints,
    const CvSize &imgDim,
    const CvMat *lM, const CvMat *rM,
    const CvMat *lD, const CvMat *rD,
    const CvMat *F,
    CvMat *mxLeft, CvMat *myLeft,
    CvMat *mxRight, CvMat *myRight)
{
    double c_lH[3][3], c_rH[3][3];
    double c_iM[3][3], c_R1[3][3], c_R2[3][3], c_TR[3][3];

    CvMat v_iM = cvMat(3, 3, CV_64F, c_iM);
    CvMat v_lH = cvMat(3, 3, CV_64F, c_lH);
    CvMat v_rH = cvMat(3, 3, CV_64F, c_rH);
    CvMat v_R1 = cvMat(3, 3, CV_64F, c_R1);
    CvMat v_R2 = cvMat(3, 3, CV_64F, c_R2);
    CvMat v_TR = cvMat(3, 3, CV_64F, c_TR);

    cvStereoRectifyUncalibrated(leftPoints, rightPoints, F,
                                imgDim, &v_lH, &v_rH);
    cvInvert(lM, &v_iM);
    cvMatMul(&v_iM, &v_lH, &v_TR);
    cvMatMul(&v_TR, lM, &v_R1);

    cvInvert(rM, &v_iM);
    cvMatMul(&v_iM, &v_rH, &v_TR);
    cvMatMul(&v_TR, rM, &v_R2);

    cvInitUndistortRectifyMap(lM, lD, &v_R1,
                              lM, mxLeft, myLeft);

    cvInitUndistortRectifyMap(rM, rD, &v_R2,
                              rM, mxRight, myRight);
}

bool RectificationCV::update()
{
    WriteGuard<ReadWritePipe<FloatImage> > lwg(m_wlpipe), rwg(m_wrpipe);

    IplImage *limg_c, *rimg_c;

    if ( !m_rlpipe->read(&limg_c)  )
    {
        return false;
    }

    if ( !m_rrpipe->read(&rimg_c) )
    {
        //limg.dispose();
        return false;
    }

    CvSize imgSz = cvSize(std::max(limg_c->width, rimg_c->width),
                        std::max(limg_c->height, rimg_c->height));
    CvSize calibPatternDim = cvSize(8, 7);

    CvMat *leftPoints, *rightPoints;
    findCornersPoints(limg_c, rimg_c, imgSz, &leftPoints, &rightPoints);

    double c_F[9];
    CvMat v_F = cvMat(3, 3, CV_64F, c_F);

    if ( m_camsDesc.hasFundamentalMatrix() )
    {
        const int fmCount = cvFindFundamentalMat(
            leftPoints, rightPoints, &v_F);
        TDV_LOG(deb).printf("Fundamental matrices found = %d", fmCount);
    }
    else
    {
        memcpy(c_F, m_camsDesc.fundamentalMatrix(), sizeof(double)*9);
    }

    const CameraParameters &lParms = m_camsDesc.leftCamera();
    const CameraParameters &rParms = m_camsDesc.rightCamera();

    double c_lM[9], c_rM[9];
    CvMat v_lM = cvMat(3, 3, CV_64F, c_lM);
    CvMat v_rM = cvMat(3, 3, CV_64F, c_rM);
    memcpy(c_lM, lParms.intrinsics(), sizeof(double)*9);
    memcpy(c_rM, rParms.intrinsics(), sizeof(double)*9);

    double c_lD[5], c_rD[5];
    CvMat v_lD = cvMat(5, 1, CV_64F, c_lD);
    CvMat v_rD = cvMat(5, 1, CV_64F, c_rD);
    memcpy(c_lD, lParms.distortion(), sizeof(double)*5);
    memcpy(c_rD, rParms.distortion(), sizeof(double)*5);

    CvMat *mxLeft = cvCreateMat(imgSz.height,  imgSz.width, CV_32FC1),
        *myLeft = cvCreateMat(imgSz.height,  imgSz.width, CV_32FC1),
        *mxRight = cvCreateMat(imgSz.height,  imgSz.width, CV_32FC1),
        *myRight = cvCreateMat(imgSz.height,  imgSz.width, CV_32FC1);

    if ( m_camsDesc.hasExtrinsics() )
    {
        double c_R[9], c_T[3];
        CvMat v_R = cvMat(3, 3, CV_64F, c_R);
        CvMat v_T = cvMat(3, 1, CV_64F, c_T);
        memcpy(c_R, m_camsDesc.extrinsicsR(), sizeof(double)*9);
        memcpy(c_T, m_camsDesc.extrinsicsT(), sizeof(double)*3);

        calibratedRectify(
            &v_lM, &v_rM, &v_lD, &v_rD, &v_R, &v_T,
            &v_F, imgSz, mxLeft, myLeft, mxRight, myRight);
    }
    else
    {
        uncalibrateRectify(
            leftPoints, rightPoints,
            imgSz,
            &v_lM, &v_rM, &v_lD, &v_rD, &v_F,
            mxLeft, myLeft, mxRight, myRight);
    }

    const Dim imgDim(imgSz.width, imgSz.height);
    // Precompute map for cvRemap()
    FloatImage limout = FloatImage::CreateCPU(imgDim);
    FloatImage rimout = FloatImage::CreateCPU(imgDim);

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

    const bool bothWriten = lwg.wasWrite() && rwg.wasWrite();
    if ( !bothWriten )
    {
        lwg.finish();
        rwg.finish();
    }

    return bothWriten;
}

TDV_NAMESPACE_END
