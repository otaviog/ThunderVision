#include "cvreprojector.hpp"

TDV_NAMESPACE_BEGIN

CVReprojector::CVReprojector()
{        
    CvMat mat = cvMat(4, 4, CV_64F, m_qMatrix);
    cvSetIdentity(&mat);
}

Vec3f CVReprojector::reproject(int x, int y, float disp) const
{
    float xyd[] = { x, y, disp };
    Vec3f dst;
        
    const CvMat srcArr = cvMat(3, 1, CV_32F, xyd);
    CvMat dstArr = cvMat(3, 1, CV_32F, dst.v);
    const CvMat Q = cvMat(4, 4, CV_64F, m_qMatrix);
        
    cvPerspectiveTransform(&srcArr, &dstArr, &Q);
        
    return dst;
}

TDV_NAMESPACE_END
