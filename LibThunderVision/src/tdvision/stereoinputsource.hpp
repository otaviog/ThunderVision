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
    virtual ReadPipe<CvMat*> *leftImgOutput() = 0;
    
    virtual ReadPipe<CvMat*> *rightImgOutput() = 0;
    
    virtual void framesPerSec(float fps) = 0;
    
private:
};

TDV_NAMESPACE_END

#endif /* TDV_STEREOINPUTSOURCE_HPP */
