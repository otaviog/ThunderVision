#ifndef TDV_DILATE_HPP
#define TDV_DILATE_HPP

#include <tdvbasic/common.hpp>
#include "floatimage.hpp"
#include "workunitutil.hpp"
#include "tmpbufferimage.hpp"

TDV_NAMESPACE_BEGIN

class Dilate: public MonoWorkUnit<FloatImage>
{
public:
    Dilate();
    
protected:
    FloatImage updateImpl(FloatImage mat);
    TmpBufferImage m_erode;
};

TDV_NAMESPACE_END

#endif /* TDV_DILATE_HPP */
