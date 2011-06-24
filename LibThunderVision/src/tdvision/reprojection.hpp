#ifndef TDV_REPROJECTION_HPP
#define TDV_REPROJECTION_HPP

#include <tdvbasic/common.hpp>
#include "floatimage.hpp"
#include "reprojector.hpp"

TDV_NAMESPACE_BEGIN

class Reprojection
{
public:
    virtual void reproject(FloatImage image, CvMat *origin, 
                           Reprojector *repr) = 0;
    
private:
};

TDV_NAMESPACE_END

#endif /* TDV_REPROJECTION_HPP */
