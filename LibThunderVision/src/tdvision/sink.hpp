#ifndef TDV_SINK_HPP
#define TDV_SINK_HPP

#include <tdvbasic/common.hpp>
#include <cv.h>
#include "workunit.hpp"
#include "pipe.hpp"
#include "floatimage.hpp"

TDV_NAMESPACE_BEGIN

template<typename Type>
struct NoSink
{
    static void sink(Type tp) 
    { /*Do nothing*/ }
};

template<typename Type, typename SinkPolicy>
class Sink: public WorkUnit
{
public:    
    Sink()
    {
        m_rpipe = NULL;
    }
    
    void input(ReadPipe<FloatImage> *rpipe)
    {
        m_rpipe = rpipe;
    }
        
    bool update();
    
private:
    ReadPipe<Type> *m_rpipe;
};

template<typename Type, typename SinkPolicy>
bool Sink<Type, SinkPolicy>::update()
{
    Type sinkData;
    if ( m_rpipe->read(&sinkData) )
    {
        SinkPolicy::sink(sinkData);
        return true;
    }

    return false;
}

struct FloatImageSinkPol
{
    static void sink(FloatImage img) 
    { 
        img.dispose();
    }    
};

struct IplImageSinkPol
{
    static void sink(IplImage *img)
    {
        cvReleaseImage(&img);
    }
};

typedef Sink<FloatImage, FloatImageSinkPol> FloatImageSink;
typedef Sink<IplImage*, IplImageSinkPol> IplImageSink;

TDV_NAMESPACE_END

#endif /* TDV_SINK_HPP */
