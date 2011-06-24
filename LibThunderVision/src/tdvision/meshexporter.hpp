#ifndef TDV_MESHEXPORTER_HPP
#define TDV_MESHEXPORTER_HPP

#include <tdvbasic/common.hpp>
#include "gridglmesh.hpp"

TDV_NAMESPACE_BEGIN

class MeshExporter
{
public:
    virtual void exportMesh(const GridGLMesh &mesh) = 0;
};

TDV_NAMESPACE_END

#endif /* TDV_MESHEXPORTER_HPP */

