#include "cvreprojector.hpp"

TDV_NAMESPACE_BEGIN

CVReprojector::CVReprojector()
{        
    CvMat mat = cvMat(4, 4, CV_32F, m_qMatrix);
    cvSetIdentity(&mat);
}

ud::Vec3f CVReprojector::reproject(int x, int y, float disp, const Dim &imgDim) const
{
    float xyd[] = { x, y, disp };
    ud::Vec3f dst;
        
    const CvMat srcArr = cvMat(1, 1, CV_32FC3, xyd);
    CvMat dstArr = cvMat(1, 1, CV_32FC3, dst.v);
    const CvMat Q = cvMat(4, 4, CV_32F, m_qMatrix);
        
    cvPerspectiveTransform(&srcArr, &dstArr, &Q);
        
    return dst;
}

TDV_NAMESPACE_END
