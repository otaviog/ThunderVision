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

CameraParameters::CameraParameters()
{
    CvMat M = intrinsecs();
    CvMat D = distorsion();
    cvSetIdentity(&M);
    cvSetIdentity(&D);
}

ChessboardPattern::ChessboardPattern()
{
    m_dim = cvSize(8, 8);    
    m_squareSize = 1.0f;        
}

void ChessboardPattern::generateObjectPoints(std::vector<CvPoint3D32f> &objPts)
{
    for (size_t i=0; i<m_dim.height; i++)
    {
        for (size_t j=0; j<m_dim.width; j++)
        {
            objPts[i*m_dim.width + j] = 
                cvPoint3D32f(i*m_squareSize, j*m_squareSize, 0);
        }
    }
}

Calibration::Calibration(size_t numFrames)
{
    m_numFrames = numFrames;
    m_frameCount = 0;
}

void Calibration::chessPattern(const ChessboardPattern &cbpattern)
{    
    m_cbpattern = cbpattern;
    
    const size_t totalPoints = cbpattern.totalCorners()*m_numFrames;    
    m_lPoints.resize(totalPoints);
    m_rPoints.resize(totalPoints);    
    m_objPoints.resize(cbpattern.totalCorners());        

    cbpattern.generateObjectPoints(m_objPoints);
    
    for (size_t i=1; i<m_numFrames; i++)
    {
        std::copy(m_objPoints.begin(),
                  m_objPoints.end(),
                  m_objPoints.begin() + cbpattern.totalCorners()*i);
    }
}

IplImage* Calibration::updateChessboardCorners(IplImage *limg, IplImage *rimg)
{    
    boost::scoped_array<CvPoint2D32f> leftPoints(
        new CvPoint2D32f[totalCorners]);
    
    boost::scoped_array<CvPoint2D32f> rightPoints(
        new CvPoint2D32f[totalCorners]);

    int leftPointsCount, rightPointsCount;
    leftPointsCount = rightPointsCount = totalCorners;
        
    cvFindChessboardCorners(limg, m_cbpattern.dim(), leftPoints.begin(),
                            &leftPointsCount);
    
    cvFindChessboardCorners(rimg, m_cbpattern.dim(), rightPoints.begin(),
                            &rightPointsCount);

    if ( leftPointsCount < totalCorners 
         && rightPointsCount < totalCorners )
    {
        return NULL;
    }
        
    IplImage *dtcImg = cvCreateImage(cvGetSize(limg), 8, 3);
    int chessboardFound = 0;
    
    cvCvtColor(limg, dtcImg, CV_GRAY2BGR);
    cvDrawChessboardCorners(dtcImg, m_cbpattern.dim(), leftPoints.get(),
                            leftPointsCount, chessboardFound);
    
    return dtcImg;        
}

void Calibration::updateCalibration()
{
    const size_t totalPoints = m_frameCount*m_cbpattern.totalCorners();            
    
    CvMat v_objectPoints = cvMat(1, N, CV_32FC3, &m_objPoints[0]);
    CvMat v_leftPoints = cvMat(1, N, CV_32FC3, m_lPoints[0]);
    CvMat v_rightPoints = cvMat(1, N, CV_32FC3, m_rightPoints[0]);
    CvMat v_npoints = cvMat(1, npoints.size(), CV_32S, &npoints[0]);
    
    double leftM[3][3], rightM[3][3], leftD[5], rightD[5],
        R[3][3], T[3];

    m_calib
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
        CV_CALIB_FIX_ASPECT_RATIO + CV_CALIB_ZERO_TANGENT_DIST + CV_CALIB_SAME_FOCAL_LENGTH);        
    
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
    
    IplImage *patternDetectPrg = updateChessboardCorners(limg, rimg);
        
    if ( patternDetectPrg == NULL )
    {
        return true;
    }
    
    wg.write(patternDetectPrg);                
    m_frameCount = (m_frameCount + 1) % m_numFrames;
            
    updateCalibration();
    
    return true;
}

TDV_NAMESPACE_END
