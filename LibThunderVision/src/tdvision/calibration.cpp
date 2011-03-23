#include <iostream>
#include <boost/scoped_array.hpp>
#include "calibration.hpp"

TDV_NAMESPACE_BEGIN

static IplImage* convertTo8UGray(IplImage *img)
{
    IplImage *grayImg = cvCreateImage(
        cvGetSize(img), IPL_DEPTH_8U, 1);

    cvCvtColor(img, grayImg, CV_RGB2GRAY);
    return grayImg;
}


Calibration::Calibration(size_t numFrames)
{
    m_numFrames = numFrames;
    chessPattern(ChessboardPattern());
    m_observer = NULL;
}

void Calibration::chessPattern(const ChessboardPattern &cbpattern)
{
    m_cbpattern = cbpattern;

    const size_t totalPoints = cbpattern.totalCorners()*m_numFrames;
    m_lPoints.resize(totalPoints);
    m_rPoints.resize(totalPoints);
    m_objPoints.resize(totalPoints);

    cbpattern.generateObjectPoints(m_objPoints);

    for (size_t i=1; i<m_numFrames; i++)
    {
        std::copy(m_objPoints.begin(),
                  m_objPoints.begin() + cbpattern.totalCorners(),
                  &m_objPoints[cbpattern.totalCorners()*i]);
    }

    m_avalFrames = 0;
    m_currFrame = 0;
}

IplImage* Calibration::updateChessboardCorners(
    const IplImage *limg, const IplImage *rimg)
{
    const size_t totalCorners = m_cbpattern.totalCorners();
    boost::scoped_array<CvPoint2D32f> leftPoints(
        new CvPoint2D32f[totalCorners]);

    boost::scoped_array<CvPoint2D32f> rightPoints(
        new CvPoint2D32f[totalCorners]);

    int leftPointsCount, rightPointsCount;
    leftPointsCount = rightPointsCount = totalCorners;

    cvFindChessboardCorners(limg, m_cbpattern.dim(), leftPoints.get(),
                            &leftPointsCount);

    cvFindChessboardCorners(rimg, m_cbpattern.dim(), rightPoints.get(),
                            &rightPointsCount);

    if ( leftPointsCount < totalCorners
         && rightPointsCount < totalCorners )
    {
        std::cout<<"No points: "
                 <<leftPointsCount<<' '
                 <<rightPointsCount<<std::endl;
        return NULL;
    }

    std::copy(leftPoints.get(), leftPoints.get() + totalCorners,
              m_lPoints.begin() + totalCorners*m_currFrame);

    std::copy(rightPoints.get(), rightPoints.get() + totalCorners,
              m_rPoints.begin() + totalCorners*m_currFrame);

    IplImage *dtcImg = cvCreateImage(cvGetSize(limg), 8, 3);
    int chessboardFound = 0;

    cvCvtColor(limg, dtcImg, CV_GRAY2BGR);
    cvDrawChessboardCorners(dtcImg, m_cbpattern.dim(), leftPoints.get(),
                            leftPointsCount, chessboardFound);

    return dtcImg;
}

void Calibration::updateCalibration(const CvSize &imgSize)
{
    const size_t totalPoints = m_avalFrames*m_cbpattern.totalCorners();

    CvMat v_objectPoints = cvMat(1, totalPoints, CV_32FC3, &m_objPoints[0]);
    CvMat v_leftPoints = cvMat(1, totalPoints, CV_32FC2, &m_lPoints[0]);
    CvMat v_rightPoints = cvMat(1, totalPoints, CV_32FC2, &m_rPoints[0]);

    std::vector<int> npoints;
    npoints.resize(m_avalFrames, m_cbpattern.totalCorners());
    CvMat v_npoints = cvMat(1, m_avalFrames, CV_32S, &npoints[0]);
    
    double c_lM[9], c_rM[9], 
        c_lD[5] = {0, 0, 0, 0, 0},
        c_rD[5] = {0, 0, 0, 0, 0},
            c_F[9], c_R[9], 
                c_T[3] = {0, 0, 0};
    
    CvMat v_leftM = cvMat(3, 3, CV_64F, c_lM),
        v_rightM = cvMat(3, 3, CV_64F, c_rM),
        v_leftD = cvMat(1, 5, CV_64F, c_lD),
        v_rightD = cvMat(1, 5, CV_64F, c_rD),
        v_R = cvMat(3, 3, CV_64F, c_R),
        v_T = cvMat(3, 1, CV_64F, c_T),
        v_F = cvMat(3, 3, CV_64F, c_F);

    cvSetIdentity(&v_leftM);
    cvSetIdentity(&v_rightM);
    cvSetIdentity(&v_F);
    cvSetIdentity(&v_R);    
    
    cvStereoCalibrate(
        &v_objectPoints, &v_leftPoints, &v_rightPoints,
        &v_npoints,
        &v_leftM, &v_leftD,
        &v_rightM, &v_rightD,
        imgSize,
        &v_R, &v_T, NULL, &v_F,
        cvTermCriteria(CV_TERMCRIT_ITER+
                       CV_TERMCRIT_EPS, 100, 1e-5),
        CV_CALIB_FIX_ASPECT_RATIO 
        + CV_CALIB_ZERO_TANGENT_DIST + CV_CALIB_SAME_FOCAL_LENGTH);
    
    m_camDesc.leftCamera().intrinsics(c_lM);
    m_camDesc.leftCamera().distortion(c_lD[0], c_lD[1], c_lD[2], c_lD[3], c_lD[4]);
    
    m_camDesc.rightCamera().intrinsics(c_rM);
    m_camDesc.rightCamera().distortion(c_rD[0], c_rD[1], c_rD[2], c_rD[3], c_rD[4]);
    
    m_camDesc.fundamentalMatrix(c_F);
    m_camDesc.extrinsics(c_R, c_T);
}

bool Calibration::update()
{
    WriteGuard<ReadWritePipe<IplImage*> > wg(m_dipipe);

    IplImage *limgOrigin, *rimgOrigin;

    if ( !m_rlpipe->read(&limgOrigin) || !m_rrpipe->read(&rimgOrigin) )
        return false;

    if ( m_sinkLeft )
        cvReleaseImage(&limgOrigin);

    if ( m_sinkRight )
        cvReleaseImage(&rimgOrigin);

    IplImage *limg = convertTo8UGray(limgOrigin),
        *rimg = convertTo8UGray(rimgOrigin);

    const CvSize imgSz = cvGetSize(limg);

    IplImage *patternDetectPrg = updateChessboardCorners(
        limg, rimg);

    cvReleaseImage(&rimg);

    if ( patternDetectPrg == NULL )
    {
        wg.write(limg);
        return true;
    }

    cvReleaseImage(&limg);    

    m_currFrame = (m_currFrame + 1) % m_numFrames;
    m_avalFrames = std::min(m_avalFrames + 1, m_numFrames);

    updateCalibration(imgSz);

    if ( m_observer != NULL )
    {
        m_observer->calibrationUpdate(*this);
    }

    wg.write(patternDetectPrg);
    
    return true;
}

TDV_NAMESPACE_END