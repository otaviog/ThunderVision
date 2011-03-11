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
    void input(ReadPipe<FloatImage> *rpipe)
    {
        m_rpipe = rpipe;
    }
        
    bool update();
    
private:
    ReadPipe<FloatImage> *m_rpipe;
};

TDV_NAMESPACE_END

#endif /* TDV_IMAGESINK_HPP */
