#ifndef TDV_IMAGEREADER_HPP
#define TDV_IMAGEREADER_HPP

#include <tdvbasic/common.hpp>
#include <string>
#include <vector>
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
        : m_infilename(filename)
    {
        workName("Image Reader");
        m_mode = mode;
        m_cImg = 0;
        loadImages();
    }
        
    ReadPipe<CvMat*>* output()
    {
        return &m_wpipe;
    }
    
    void reset()
    {
        m_cImg = 0;
        m_wpipe.reset();
    }
    
    bool update();

private:
    void loadImages();
        
    ReadWritePipe<CvMat*> m_wpipe;
    std::string m_infilename;
    std::vector<std::string> m_filenames;
    size_t m_cImg;
    Mode m_mode;
};

TDV_NAMESPACE_END

#endif /* TDV_IMAGEREADER_HPP */
