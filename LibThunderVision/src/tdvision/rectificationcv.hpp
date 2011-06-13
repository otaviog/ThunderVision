#ifndef TDV_RECTIFICATIONCV_HPP
#define TDV_RECTIFICATIONCV_HPP

#include <tdvbasic/common.hpp>
#include <cv.h>
#include "workunit.hpp"
#include "pipe.hpp"
#include "floatimage.hpp"
#include "tmpbufferimage.hpp"
#include "camerasdesc.hpp"
#include "misc.hpp"
#include "cvreprojector.hpp"

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


class RectificationCV: public WorkUnit, public Reprojector
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
    
    ReadPipe<CvMat*>* colorLeftImgOutput()
    {
        return &m_wclpipe;
    }
    
    ReadPipe<CvMat*>* colorRightImgOutput()
    {
        return &m_wcrpipe;
    }

    const CamerasDesc& camerasDesc() const
    {
        return m_camsDesc;
    }

    void camerasDesc(const CamerasDesc &desc)
    {
        m_camsDesc = desc;
        m_camsDescChanged = true;
    }
    
    ud::Vec3f reproject(int x, int y, float disp, const Dim &imgDim) const
    {
        return m_repr.reproject(x, y, disp, imgDim);
    };    
    
    
    void enableColorRemap() 
    {
        m_enableColorRemap = true;
    }
    
    void disableColorRemap()
    {
        m_enableColorRemap = false;
    }
    
    bool update();
    
private:
    void updateRectification(CvMat *limg8u, CvMat *rimg8u);

    void findCorners(const CvMat *img, CvPoint2D32f *leftCorners, 
                     int *cornerCount, CvMat *eigImage, CvMat *tmpImage);

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
    ReadWritePipe<CvMat*> m_wclpipe, m_wcrpipe;
    CamerasDesc m_camsDesc;
    bool m_camsDescChanged, m_enableColorRemap;

    ConjugateCorners m_conjCorners;
    
    TmpBufferImage m_limg32f, m_rimg32f,
        m_limg8u, m_rimg8u,
        m_mxLeft, m_myLeft, m_mxRight, m_myRight;
    
    misc::Conv8UC3To32FC1 m_convTo32f;
    misc::Conv8UC3To32FC1Hsv m_convTo32fHsv;
    
    CVReprojector m_repr;
};

TDV_NAMESPACE_END

#endif /* TDV_RECTIFICATIONCV_HPP */
