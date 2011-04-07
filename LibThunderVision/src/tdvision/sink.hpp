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

    static void incrRef(Type tp)
    {
    }
};

template<typename Type, typename SinkPolicy>
class Sink: public WorkUnit
{
public:    
    Sink()
    {
        m_rpipe = NULL;
    }
    
    void input(ReadPipe<Type> *rpipe)
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
    static void incrRef(FloatImage img);
    
    static void sink(FloatImage img);
};

struct IplImageSinkPol
{
    static void sink(IplImage *img)
    {
        cvReleaseImage(&img);
    }
};

struct CvMatSinkPol
{    
    static void incrRef(CvMat *mat);
    
    static void sink(CvMat *mat);
};

typedef Sink<FloatImage, FloatImageSinkPol> FloatImageSink;
typedef Sink<IplImage*, IplImageSinkPol> IplImageSink;
typedef Sink<CvMat*, CvMatSinkPol> CvMatSink;

template<typename Type>
struct SinkTraits
{    

};

template<>
struct SinkTraits<FloatImage>
{
    typedef FloatImageSinkPol Sinker;
};

template<>
struct SinkTraits<IplImage*>
{
    typedef IplImageSinkPol Sinker;
};

template<>
struct SinkTraits<CvMat*>
{
    typedef CvMatSinkPol Sinker;
};

TDV_NAMESPACE_END

#endif /* TDV_SINK_HPP */
