#ifndef TDV_IMAGEWRITERWU_HPP
#define TDV_IMAGEWRITERWU_HPP

#include <tdvbasic/common.hpp>
#include "floatimage.hpp"
#include "pipe.hpp"
#include "workunit.hpp"

TDV_NAMESPACE_BEGIN

class ImageWriter: public WorkUnit
{
public:
    ImageWriter(const std::string &filename)       
        : m_filename(filename)
    {
        workName("Image Writer");
    }
    
    void input(ReadPipe<FloatImage> *rpipe)
    { 
        m_rpipe = rpipe;
    }
    
    ReadPipe<FloatImage>* output()
    { 
        return &m_wpipe;
    }

    bool update();
        
private:
    ReadPipe<FloatImage> *m_rpipe;
    ReadWritePipe<FloatImage, FloatImage> m_wpipe;
    
    std::string m_filename;
};

TDV_NAMESPACE_END

#endif /* TDV_IMAGEWRITERWU_HPP */
