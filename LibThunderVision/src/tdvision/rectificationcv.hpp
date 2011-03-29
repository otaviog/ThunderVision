#ifndef TDV_RECTIFICATIONCV_HPP
#define TDV_RECTIFICATIONCV_HPP

#include <tdvbasic/common.hpp>
#include "workunit.hpp"
#include "pipe.hpp"
#include "floatimage.hpp"
#include "camerasdesc.hpp"

TDV_NAMESPACE_BEGIN

class RectificationCV: public WorkUnit
{
public:
    void leftImgInput(ReadPipe<IplImage*> *rlpipe)
    {
        m_rlpipe = rlpipe;
    }
    
    void rightImgInput(ReadPipe<IplImage*> *rrpipe)
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
    void findCorners(const IplImage *img, CvPoint2D32f *leftCorners, int *cornerCount,
                     IplImage *eigImage, IplImage *tmpImage);
    
    size_t findCornersPoints(const IplImage *limg_c, const IplImage *rimg_c, 
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

    ReadPipe<IplImage*> *m_rlpipe, *m_rrpipe;
    ReadWritePipe<FloatImage> m_wlpipe, m_wrpipe;
    CamerasDesc m_camsDesc;
};

TDV_NAMESPACE_END

#endif /* TDV_RECTIFICATIONCV_HPP */
