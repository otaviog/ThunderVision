#ifndef TDV_IPLIMAGESINK_HPP
#define TDV_IPLIMAGESINK_HPP

#include <tdvbasic/common.hpp>
#include <cv.h>
#include "workunit.hpp"
#include "pipe.hpp"

TDV_NAMESPACE_BEGIN

class IplImageSink: public WorkUnit
{
public:
    IplImageSink()
    {
        workName("IplImageSink");        
    }

    void input(ReadPipe<IplImage*> *rpipe)
    {
        m_rpipe = rpipe;
    }
    
    void process();
    
private:
    ReadPipe<IplImage*> *m_rpipe;
};

TDV_NAMESPACE_END

#endif /* TDV_IPLIMAGESINK_HPP */
