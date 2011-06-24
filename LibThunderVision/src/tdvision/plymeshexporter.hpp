#ifndef TDV_PLAYMESHEXPORTER_HPP
#define TDV_PLAYMESHEXPORTER_HPP

#include <string>
#include <tdvbasic/common.hpp>
#include "meshexporter.hpp"

TDV_NAMESPACE_BEGIN

class PLYMeshExporter: public MeshExporter
{
public:
    PLYMeshExporter(const std::string &name)
        : m_filename(name)
    { }
    
    void exportMesh(const GridGLMesh &mesh);
    
    void filename(const std::string &fname)
    {
        m_filename = fname;
    }
    
    const std::string& filename() const
    {
        return m_filename;
    }
    
private:
    std::string m_filename;    
};

TDV_NAMESPACE_END

#endif /* TDV_PLAYMESHEXPORTER_HPP */
