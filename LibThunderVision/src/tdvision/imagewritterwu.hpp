#ifndef TDV_IMAGEWRITTERWU_HPP
#define TDV_IMAGEWRITTERWU_HPP

#include <tdvbasic/common.hpp>
#include "floatimage.hpp"
#include "typedworkunit.hpp"

TDV_NAMESPACE_BEGIN

class ImageWritterWU: public TypedWorkUnit<FloatImage, FloatImage>
{
public:
    ImageWritterWU(const std::string &filename)
        : TypedWorkUnit<FloatImage, FloatImage>("Image Writter"),
          m_filename(filename)
    {
    }
    
    void input(ReadPipeType *rpipe)
    { 
        m_rpipe = rpipe;
    }
    
    void output(WritePipeType *wpipe)
    { 
        m_wpipe = wpipe;
    }

    void process();
        
private:
    ReadPipeType *m_rpipe;
    WritePipeType *m_wpipe;
    
    std::string m_filename;
};

TDV_NAMESPACE_END

#endif /* TDV_IMAGEWRITTERWU_HPP */
