#include "misc.hpp"

TDV_NAMESPACE_BEGIN

namespace misc
{
    CvMat* create8UGray(const CvArr *src)
    {
        CvSize sz(cvGetSize(src));
        CvMat *dst = cvCreateMat(sz.height, sz.width, CV_8U);
        cvCvtColor(src, dst, CV_RGB2GRAY);
        return dst;
    }
}

TDV_NAMESPACE_END
