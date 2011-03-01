#ifndef TDV_IMAGEREADERWU_HPP
#define TDV_IMAGEREADERWU_HPP

#include <tdvbasic/common.hpp>
#include "floatimage.hpp"
#include "typedworkunit.hpp"

TDV_NAMESPACE_BEGIN

class ImageReaderWU: public TypedWorkUnit<FloatImage, FloatImage>
{
public:    
    ImageReaderWU(const std::string &filename)
          m_filename(filename)
    {
        workName("Image Reader");
        m_wpipe = NULL;
    }
    
    void input(ReadPipeType *rpipe)
    { }
    
    void output(WritePipeType *wpipe)
    {
        m_wpipe = wpipe;
    }

    void process();
        
    template<typename WorkUnitType>
    void connect(WorkUnitType *nextWU)
    {
        m_wpipe = new ReadWritePipe<WorkUnitType;
        
    }
    
private:
    WritePipeType *m_wpipe;
    std::string m_filename;
};

TDV_NAMESPACE_END

#endif /* TDV_IMAGEREADERWU_HPP */
