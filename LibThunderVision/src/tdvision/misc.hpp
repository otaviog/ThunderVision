#ifndef TDV_MISC_HPP
#define TDV_MISC_HPP

#include <tdvbasic/common.hpp>
#include <cv.h>

TDV_NAMESPACE_BEGIN

namespace misc
{
    CvMat* create8UGray(const CvArr *src); 
}

TDV_NAMESPACE_END

#endif /* TDV_MISC_HPP */
