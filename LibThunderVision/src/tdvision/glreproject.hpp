#ifndef TDV_GLREPROJECTION_HPP
#define TDV_GLREPROJECTION_HPP

#include <tdvbasic/common.hpp>
#include "reprojector.hpp"

TDV_NAMESPACE_BEGIN

class GLReprojection: public Reprojection
{
public:
    GLReproject();
    
    void reproject(FloatImage image, Reprojector *repr);
    
    void draw();    
    
private:
    
    
};

TDV_NAMESPACE_END

#endif /* TDV_GLREPROJECT_HPP */
