#ifndef TDV_STEREOINPUTSOURCE_HPP
#define TDV_STEREOINPUTSOURCE_HPP

#include <tdvbasic/common.hpp>
#include <cv.h>
#include "processgroup.hpp"
#include "pipe.hpp"

TDV_NAMESPACE_BEGIN

class StereoInputSource: public ProcessGroup
{
public:
    virtual ReadPipe<IplImage*> *leftImgOutput() = 0;
    
    virtual ReadPipe<IplImage*> *rightImgOutput() = 0;
    
private:
};

TDV_NAMESPACE_END

#endif /* TDV_STEREOINPUTSOURCE_HPP */
