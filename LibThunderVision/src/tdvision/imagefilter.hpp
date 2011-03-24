#ifndef TDV_IMAGEFILTER_HPP
#define TDV_IMAGEFILTER_HPP

#include <tdvbasic/common.hpp>

TDV_NAMESPACE_BEGIN

class ImageFilter: public WorkUnit
{
public:
    virtual void input(ReadPipe<FloatImage> *rpipe) = 0;
    
    virtual ReadPipe<FloatImage>* output() = 0;
    
private:
};

TDV_NAMESPACE_END

#endif /* TDV_IMAGEFILTER_HPP */
