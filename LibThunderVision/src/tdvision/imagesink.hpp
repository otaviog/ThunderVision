#ifndef TDV_IMAGESINK_HPP
#define TDV_IMAGESINK_HPP

#include <tdvbasic/common.hpp>
#include "workunit.hpp"
#include "floatimage.hpp"
#include "pipe.hpp"

TDV_NAMESPACE_BEGIN

class ImageSink: public WorkUnit
{
public:
    ImageSink()    
    {
        workName("FloatImage Sink");
    }
    
    void input(ReadPipe<FloatImage> *rpipe)
    {
        m_rpipe = rpipe;
    }
        
    void process();
    
private:
    ReadPipe<FloatImage> *m_rpipe;
};

TDV_NAMESPACE_END

#endif /* TDV_IMAGESINK_HPP */
