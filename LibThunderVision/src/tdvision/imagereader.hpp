#ifndef TDV_IMAGEREADER_HPP
#define TDV_IMAGEREADER_HPP

#include <tdvbasic/common.hpp>
#include "floatimage.hpp"
#include "pipe.hpp"
#include "workunit.hpp"

TDV_NAMESPACE_BEGIN

class ImageReader: public WorkUnit
{
public:    
    ImageReader(const std::string &filename)
        : m_filename(filename)
    {
        workName("Image Reader");
    }
        
    ReadPipe<FloatImage>* output()
    {
        return &m_wpipe;
    }
    
    bool update();

private:
    ReadWritePipe<FloatImage, FloatImage> m_wpipe;
    std::string m_filename;
};

TDV_NAMESPACE_END

#endif /* TDV_IMAGEREADER_HPP */
