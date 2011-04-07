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
    enum Mode
    {
        Directory,
        File
    };
        
    ImageReader(const std::string &filename, Mode mode = File)
        : m_filename(filename)
    {
        workName("Image Reader");
        m_mode = mode;
    }
        
    ReadPipe<CvMat*>* output()
    {
        return &m_wpipe;
    }
    
    bool update();

private:
    void loadImages();
    
    
    ReadWritePipe<CvMat*> m_wpipe;
    std::string m_filename;
    Mode m_mode;
};

TDV_NAMESPACE_END

#endif /* TDV_IMAGEREADER_HPP */
