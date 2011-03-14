#ifndef TDV_MESHGENRATOR_HPP
#define TDV_MESHGENRATOR_HPP

#include <tdvbasic/common.hpp>
#include "workunit.hpp"

TDV_NAMESPACE_BEGIN

class MeshGenerator: public WorkUnit
{
public:        
    bool update();
    
private:
    Mesh
};

TDV_NAMESPACE_END

#endif /* TDV_MESHGENRATOR_HPP */
