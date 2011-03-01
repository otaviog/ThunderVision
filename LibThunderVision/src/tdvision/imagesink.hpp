#ifndef TDV_IMAGESINK_HPP
#define TDV_IMAGESINK_HPP

#include <tdvbasic/common.hpp>
#include "typedworkunit.hpp"
#include "floatimage.hpp"

TDV_NAMESPACE_BEGIN

class ImageSink: public TypedWorkUnit<FloatImage, FloatImage>
{
public:
    ImageSink()
        : TypedWorkUnit<FloatImage, FloatImage>("Image sink")
    {
    }
    
    void input(ReadPipeType *rpipe)
    {
        m_rpipe = rpipe;
    }
    
    void output(WritePipeType *wpipe)
    { }
    
    void process();
    
private:
    ReadPipeType *m_rpipe;
};

TDV_NAMESPACE_END

#endif /* TDV_IMAGESINK_HPP */
