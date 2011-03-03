#ifndef TDV_IMAGEREADERWU_HPP
#define TDV_IMAGEREADERWU_HPP

#include <tdvbasic/common.hpp>
#include "floatimage.hpp"
#include "pipe.hpp"
#include "workunit.hpp"

TDV_NAMESPACE_BEGIN

class ImageReaderWU: public WorkUnit
{
public:    
    ImageReaderWU(const std::string &filename)
        : m_filename(filename)
    {
        workName("Image Reader");
    }
        
    ReadPipe<FloatImage>* output()
    {
        return &m_wpipe;
    }

    void process();

private:
    ReadWritePipe<FloatImage, FloatImage> m_wpipe;
    std::string m_filename;
};

TDV_NAMESPACE_END

#endif /* TDV_IMAGEREADERWU_HPP */
