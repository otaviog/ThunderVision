#ifndef TDV_RECTIFICATIONCV_HPP
#define TDV_RECTIFICATIONCV_HPP

#include <tdvbasic/common.hpp>
#include "workunit.hpp"
#include "pipe.hpp"
#include "floatimage.hpp"

TDV_NAMESPACE_BEGIN

class RectificationCV: public WorkUnit
{
public:
    void leftImgInput(ReadPipe<FloatImage> *rlpipe)
    {
        m_rlpipe = rlpipe;
    }
    
    void rightImgInput(ReadPipe<FloatImage> *rrpipe)
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
    
    bool update();
    
private:
    void findCorners(IplImage *img, CvPoint2D32f *leftCorners, int *cornerCount,
                     IplImage *eigImage, IplImage *tmpImage);
    
    ReadPipe<FloatImage> *m_rlpipe, *m_rrpipe;
    ReadWritePipe<FloatImage> m_wlpipe, m_wrpipe;
};

TDV_NAMESPACE_END

#endif /* TDV_RECTIFICATIONCV_HPP */
