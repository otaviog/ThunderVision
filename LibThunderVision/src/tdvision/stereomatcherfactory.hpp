#ifndef TDV_STEREOMATCHERFACTORY_HPP
#define TDV_STEREOMATCHERFACTORY_HPP

#include <tdvbasic/common.hpp>

TDV_NAMESPACE_BEGIN

class StereoMatcher;

class StereoMatcherFactory
{
public:
    virtual ~StereoMatcherFactory() 
    { }
    
    virtual StereoMatcher* createStereoMatcher() = 0;
};

TDV_NAMESPACE_END

#endif /* TDV_STEREOMATCHERFACTORY_HPP */
