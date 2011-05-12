#include <highgui.h>
#include <tdvbasic/log.hpp>
#include "misc.hpp"
#include "sink.hpp"
#include "rectificationcv.hpp"

#ifdef SHOW_DEB_IMGS
#define showImage(S, I) cvShowImage(S, I)
#define waitKey(V) cvWaitKey(V)
#else
#define showImage(S, I)
#define waitKey(V)
#endif

TDV_NAMESPACE_BEGIN

ConjugateCorners::ConjugateCorners()
    : m_eigImage(CV_8U),
      m_tmpImage(CV_8U)
{    
}

void ConjugateCorners::findCorners(const CvMat *img, CvPoint2D32f *corners,
                                   int *cornerCount,
                                   CvMat *eigImage, CvMat *tmpImage)
{
    static const size_t CORNER_SEARCH_WIN_DIM = 9;

    cvGoodFeaturesToTrack(img, eigImage, tmpImage,
                          corners, cornerCount,
                          0.05, 5.0);

    cvFindCornerSubPix(img, corners, *cornerCount,
                       cvSize(CORNER_SEARCH_WIN_DIM, CORNER_SEARCH_WIN_DIM),
                       cvSize(-1, -1),
                       cvTermCriteria(
                           CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
}

void ConjugateCorners::updateConjugates(const CvMat *leftImg, const CvMat *rightImg)
{
    static const size_t MAX_CORNERS = 15;
    CvPoint2D32f leftCorners[MAX_CORNERS], rightCorners[MAX_CORNERS];
    int leftCornerCount = MAX_CORNERS, rightCornerCount = MAX_CORNERS;

    CvMat *eigImage = m_eigImage.getImage(getEigSize(leftImg, rightImg));
    CvMat *tmpImage = m_tmpImage.getImage(getTmpSize(leftImg, rightImg));

    findCorners(leftImg, leftCorners, &leftCornerCount,
                eigImage, tmpImage);
    findCorners(rightImg, rightCorners, &rightCornerCount,
                eigImage, tmpImage);

    int minCornersFound = std::min(leftCornerCount, rightCornerCount);
    char status[minCornersFound];

    cvCalcOpticalFlowPyrLK(leftImg, rightImg, eigImage, tmpImage,
                           leftCorners, rightCorners, minCornersFound,
                           cvSize(9, 9), 1, status, NULL,
                           cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,
                                          20, 0.03),
                           CV_LKFLOW_INITIAL_GUESSES);


    m_lPoints.resize(minCornersFound);
    m_rPoints.resize(minCornersFound);

    for (int i=0; i<minCornersFound; i++)
    {
        m_lPoints[i].x = leftCorners[i].x;
        m_lPoints[i].y = leftCorners[i].y;

        m_rPoints[i].x = rightCorners[i].x;
        m_rPoints[i].y = rightCorners[i].y;
    }
}

RectificationCV::RectificationCV()
    : m_limg32f(CV_32F), m_rimg32f(CV_32F),
      m_limg8u(CV_8U), m_rimg8u(CV_8U),
      m_mxLeft(CV_32F), m_myLeft(CV_32F),
      m_mxRight(CV_32F), m_myRight(CV_32F)
{
    m_camsDescChanged = true;
}

void RectificationCV::calibratedRectify(const CvMat *lM, const CvMat *rM,
                                        const CvMat *lD, const CvMat *rD,
                                        const CvMat *R, const CvMat *T,
                                        const CvMat *F, const CvSize &imgSize,
                                        CvMat *mxLeft, CvMat *myLeft,
                                        CvMat *mxRight, CvMat *myRight)
{
    double c_lR[9], c_rR[9], c_lP[12], c_rP[12], c_Q[16];
    CvMat v_lR = cvMat(3, 3, CV_64F, c_lR);
    CvMat v_rR = cvMat(3, 3, CV_64F, c_rR);
    CvMat v_lP = cvMat(3, 4, CV_64F, c_lP);
    CvMat v_rP = cvMat(3, 4, CV_64F, c_rP);
    CvMat v_Q = cvMat(4, 4, CV_64F, c_Q);
    
    cvSetIdentity(&v_lR);
    cvSetIdentity(&v_rR);
    cvSetIdentity(&v_lP);
    cvSetIdentity(&v_rP);

    cvStereoRectify(lM, rM, lD, rD, imgSize, R, T,
                    &v_lR, &v_rR, &v_lP, &v_rP, &v_Q, 0);
    m_repr.qmatrix(c_Q);
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

    cvSetIdentity(&v_lH);
    cvSetIdentity(&v_rH);

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
static inline CvSize maxSize(CvMat *l, CvMat *r)
{
    return cvSize(std::max(l->width, r->width),
                  std::max(l->height, r->height));
}

void RectificationCV::updateRectification(CvMat *limg8u, CvMat *rimg8u)
{
    const CvSize imgSz(maxSize(limg8u, rimg8u));

    CvMat leftPoints, rightPoints;
    double c_F[9];
    CvMat v_F = cvMat(3, 3, CV_64F, c_F);

    if ( !m_camsDesc.hasFundamentalMatrix() )
    {
        m_conjCorners.updateConjugates(limg8u, rimg8u);
        leftPoints = m_conjCorners.leftPoints();
        rightPoints = m_conjCorners.rightPoints();

        cvSetIdentity(&v_F);

        const int fmCount = cvFindFundamentalMat(
            &leftPoints, &rightPoints, &v_F);
        TDV_LOG(deb).printf("Fundamental matrices found = %d\n", fmCount);
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

    CvMat* mxLeft = m_mxLeft.getImage(imgSz);
    CvMat* myLeft = m_myLeft.getImage(imgSz);
    CvMat* mxRight = m_mxRight.getImage(imgSz);
    CvMat* myRight = m_myRight.getImage(imgSz);
    
    if ( m_camsDesc.hasExtrinsics() )
    {
        double c_R[9], c_T[3];
        CvMat v_R = cvMat(3, 3, CV_64F, c_R);
        CvMat v_T = cvMat(3, 1, CV_64F, c_T);
        memcpy(c_R, m_camsDesc.extrinsicsR(), sizeof(double)*9);
        memcpy(c_T, m_camsDesc.extrinsicsT(), sizeof(double)*3);

        calibratedRectify(
            &v_lM, &v_rM, &v_lD, &v_rD, &v_R, &v_T,
            &v_F, imgSz,
            mxLeft, myLeft, mxRight, myRight);
    }
    else
    {
        if ( m_camsDesc.hasFundamentalMatrix() )
        {
            m_conjCorners.updateConjugates(limg8u, rimg8u);
            leftPoints = m_conjCorners.leftPoints();
            rightPoints = m_conjCorners.rightPoints();
        }

        uncalibrateRectify(
            &leftPoints, &rightPoints,
            imgSz, &v_lM, &v_rM, &v_lD, &v_rD, &v_F,
            mxLeft, myLeft, mxRight, myRight);
    }    
}

bool RectificationCV::update()
{
    WriteGuard<ReadWritePipe<FloatImage> > lwg(m_wlpipe), rwg(m_wrpipe);

    CvMat *limg_c, *rimg_c;

    if ( !m_rlpipe->read(&limg_c)  )
    {
        return false;
    }

    if ( !m_rrpipe->read(&rimg_c) )
    {
         CvMatSinkPol::sink(limg_c);
         return false;
    }
        
    const CvSize imgSz(maxSize(limg_c, rimg_c));
    
    CvMat *limg32f = m_limg32f.getImage(cvGetSize(limg_c));
    CvMat *rimg32f = m_rimg32f.getImage(cvGetSize(rimg_c));
    CvMat *limg8u = m_limg8u.getImage(cvGetSize(limg_c));
    CvMat *rimg8u = m_rimg8u.getImage(cvGetSize(rimg_c));

    misc::convert8UC3To32FC1Gray(limg_c, limg32f, limg8u);
    misc::convert8UC3To32FC1Gray(rimg_c, rimg32f, rimg8u);

    showImage("OL", limg_c);
    showImage("OR", rimg_c);

    CvMatSinkPol::sink(limg_c);
    CvMatSinkPol::sink(rimg_c);

    showImage("GL", limg8u);
    showImage("GR", rimg8u);

    if ( m_camsDescChanged )
    {
        updateRectification(limg8u, rimg8u);
        m_camsDescChanged = false;
    }
    
    const Dim imgDim(imgSz.width, imgSz.height);

    FloatImage limout = FloatImage::CreateCPU(imgDim);
    FloatImage rimout = FloatImage::CreateCPU(imgDim);

    CvMat *limout_c = limout.cpuMem();
    CvMat *rimout_c = rimout.cpuMem();
    
#if 1
    cvRemap(limg32f, limout_c, m_mxLeft.get(), m_myLeft.get());
    cvRemap(rimg32f, rimout_c, m_mxRight.get(), m_myRight.get());

    showImage("L", limout_c);
    showImage("R", rimout_c);
    waitKey(0);
#else
    cvConvert(limg32f, limout_c);
    cvConvert(rimg32f, rimout_c);
#endif
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
