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

ChessboardPattern::ChessboardPattern()
{
    m_dim = cvSize(8, 8);
    m_objPoints = new CvPoint3D32f[totalCorners()];
    m_squareSize = 1.0f;
    
    for (size_t i=0; i<m_dim.height; i++)
    {
        for (size_t j=0; j<m_dim.width; j++)
        {
            m_objPoints[i*m_dim.width + j] = 
                cvPoint3D32f(i*m_squareSize, j*m_squareSize, 0);
        }
    }
}

Calibration::Calibration()
{
    
}

bool Calibration::update()
{
    WriteGuard<ReadWritePipe<IplImage*> > wg(m_dipipe);
    
    IplImage *limgOrigin, *rimgOrigin;
    
    if ( !m_rlpipe->read(&limgOrigin) || !m_rrpipe->read(&rimgOrigin) )
        return false;
    
    IplImage *limg = convertTo8UGray(limgOrigin), 
        *rimg = convertTo8UGray(rimgOrigin);

    //cvReleaseImage(&limgOrigin);
    //cvReleaseImage(&rimgOrigin);
    
    const size_t totalCorners = m_chessboard.totalCorners();
    
    boost::scoped_array<CvPoint2D32f> leftPoints(
        new CvPoint2D32f[totalCorners]);
    boost::scoped_array<CvPoint2D32f> rightPoints(
        new CvPoint2D32f[totalCorners]);
    
    int leftPointsCount, rightPointsCount;
    leftPointsCount = rightPointsCount = totalCorners;
    
    cvFindChessboardCorners(limg, m_chessboard.dim(), leftPoints.get(),
                            &leftPointsCount);    
    cvFindChessboardCorners(rimg, m_chessboard.dim(), rightPoints.get(),
                            &rightPointsCount);

    IplImage *dtcImg = cvCreateImage(cvGetSize(limg), 8, 3);
    int chessboardFound = 0;
    
    cvCvtColor(limg, dtcImg, CV_GRAY2BGR);
    cvDrawChessboardCorners(dtcImg, m_chessboard.dim(), leftPoints.get(),
                            leftPointsCount, chessboardFound);
    
    wg.write(dtcImg);

    const size_t N = 2*m_chessboard.totalCorners();
        
    std::vector<CvPoint3D32f> objectPoints;
    objectPoints.resize(N);
    std::copy(m_chessboard.objectPoints(), 
              m_chessboard.objectPoints() + m_chessboard.totalCorners(), 
              objectPoints.begin());
    std::copy(m_chessboard.objectPoints(), 
              m_chessboard.objectPoints() + m_chessboard.totalCorners(), 
              objectPoints.begin() + m_chessboard.totalCorners());
    
    std::vector<int> npoints;
    npoints.resize(2, m_chessboard.totalCorners());
    
    CvMat v_objectPoints = cvMat(1, N, CV_32FC3, &objectPoints[0]);
    CvMat v_leftPoints = cvMat(1, N, CV_32FC3, leftPoints.get());
    CvMat v_rightPoints = cvMat(1, N, CV_32FC3, rightPoints.get());
    CvMat v_npoints = cvMat(1, npoints.size(), CV_32S, &npoints[0]);
    
    double leftM[3][3], rightM[3][3], leftD[5], rightD[5],
        R[3][3], T[3];

    CvMat v_leftM = cvMat(3, 3, CV_64F, leftM),
        v_rightM = cvMat(3, 3, CV_64F, rightM),
        v_leftD = cvMat(1, 5, CV_64F, leftD),
        v_rightD = cvMat(1, 5, CV_64F, rightD),
        v_R = cvMat(3, 3, CV_64F, R),
        v_T = cvMat(3, 1, CV_64F, T);
    
    cvStereoCalibrate(
        &v_objectPoints, &v_leftPoints, &v_rightPoints, 
        &v_npoints, 
        &v_leftM, &v_leftD, 
        &v_rightM, &v_rightD,
        cvGetSize(limg), 
        &v_R, &v_T, NULL, NULL,
        cvTermCriteria(CV_TERMCRIT_ITER+
        CV_TERMCRIT_EPS, 100, 1e-5),
        CV_CALIB_FIX_ASPECT_RATIO +
        CV_CALIB_ZERO_TANGENT_DIST +
        CV_CALIB_SAME_FOCAL_LENGTH);        
    
    
    return wg.wasWrite();
}

TDV_NAMESPACE_END
