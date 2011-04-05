#ifndef TDV_RECTIFICATIONCV_HPP
#define TDV_RECTIFICATIONCV_HPP

#include <tdvbasic/common.hpp>
#include "workunit.hpp"
#include "pipe.hpp"
#include "floatimage.hpp"
#include "tmpbufferimage.hpp"
#include "camerasdesc.hpp"

TDV_NAMESPACE_BEGIN

class ConjugateCorners
{
public:    
    ConjugateCorners();
    
    void updateConjugates(const CvMat *leftImg, const CvMat *rightImg);
    
    CvMat leftPoints() 
    {
        return cvMat(1, m_lPoints.size(), CV_64FC2, &m_lPoints[0]);
    }
    
    CvMat rightPoints()
    {
        return cvMat(1, m_rPoints.size(), CV_64FC2, &m_rPoints[0]);
    }
    
private:
    void findCorners(const CvMat *img, CvPoint2D32f *corners,
                     int *cornerCount,
                     CvMat *eigImage, CvMat *tmpImage);

    CvSize getEigSize(const CvMat *leftImg, const CvMat *rightImg)
    {
        return cvSize(
            std::min(leftImg->cols, rightImg->cols) + 8,
            std::min(leftImg->rows, rightImg->rows));
    }    
    
    CvSize getTmpSize(const CvMat *leftImg, const CvMat *rightImg)
    {
        return cvSize(
            std::min(leftImg->cols, rightImg->cols),
            std::min(leftImg->rows, rightImg->rows));
    }    

    TmpBufferImage m_eigImage, m_tmpImage;    
    std::vector<CvPoint2D64f> m_lPoints, m_rPoints;
};


class RectificationCV: public WorkUnit
{
public:    
    RectificationCV();
    
    void leftImgInput(ReadPipe<CvMat*> *rlpipe)
    {
        m_rlpipe = rlpipe;
    }
    
    void rightImgInput(ReadPipe<CvMat*> *rrpipe)
    {
        m_rrpipe = rrpipe;
    }

    ReadPipe<FloatImage>* leftImgOutput()
    {
        return &m_wlpipe;
    }
    
    ReadPipe<FloatImage>* rightImgOutput()
    {
        return &m_wrpipe;
    }
    
    const CamerasDesc& camerasDesc() const
    {
        return m_camsDesc;
    }
    
    void camerasDesc(const CamerasDesc &desc)
    {
        m_camsDesc = desc;
    }
    
    bool update();
    
private:
    void findCorners(const CvMat *img, CvPoint2D32f *leftCorners, int *cornerCount,
                     CvMat *eigImage, CvMat *tmpImage);
    
    size_t findCornersPoints(const CvMat *limg_c, const CvMat *rimg_c, 
                             const CvSize &imgSize,
                             CvMat **leftPointsR, CvMat **rightPointsR);

    void calibratedRectify(const CvMat *lM, const CvMat *rM,
                           const CvMat *lD, const CvMat *rD,                                        
                           const CvMat *R, const CvMat *T, 
                           const CvMat *F, const CvSize &imgSize,
                           CvMat *mxLeft, CvMat *myLeft,
                           CvMat *mxRight, CvMat *myRight);

    void uncalibrateRectify(const CvMat *leftPoints, const CvMat *rightPoints, 
                            const CvSize &imgDim,
                            const CvMat *lM, const CvMat *rM,
                            const CvMat *lD, const CvMat *rD,
                            const CvMat *F, 
                            CvMat *mxLeft, CvMat *myLeft,
                            CvMat *mxRight, CvMat *myRight);

    ReadPipe<CvMat*> *m_rlpipe, *m_rrpipe;
    ReadWritePipe<FloatImage> m_wlpipe, m_wrpipe;
    CamerasDesc m_camsDesc;    
    
    ConjugateCorners m_conjCorners;    
    
    TmpBufferImage m_limg32f, m_rimg32f, m_limg8u, m_rimg8u;
    
};

TDV_NAMESPACE_END

#endif /* TDV_RECTIFICATIONCV_HPP */
