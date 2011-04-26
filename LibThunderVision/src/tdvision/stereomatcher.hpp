#ifndef TDV_STEREOMATCHER_HPP
#define TDV_STEREOMATCHER_HPP

#include <tdvbasic/common.hpp>
#include "processgroup.hpp"
#include "pipe.hpp"
#include "floatimage.hpp"

TDV_NAMESPACE_BEGIN

class StereoMatcher: public ProcessGroup
{
public:
    virtual void inputs(ReadPipe<FloatImage> *leftInput,
                        ReadPipe<FloatImage> *rightInput) = 0;
    
    virtual ReadPipe<FloatImage>* output() = 0;
    
    virtual std::string name() const = 0;
};

TDV_NAMESPACE_END

#endif /* TDV_STEREOMATCHER_HPP */
