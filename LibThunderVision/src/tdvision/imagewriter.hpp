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
    
    void input(ReadPipe<CvMat*> *rpipe)
    { 
        m_rpipe = rpipe;
    }
    
    void filename(const std::string &filename)
    {
        m_filename = filename;
    }

    const std::string& filename() const
    {
        return m_filename;
    }
    
    bool update();
        
private:
    ReadPipe<CvMat*> *m_rpipe;    
    std::string m_filename;
};

TDV_NAMESPACE_END

#endif /* TDV_IMAGEWRITERWU_HPP */
