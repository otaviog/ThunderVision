#include <tdvbasic/exception.hpp>
#include <fstream>
#include <boost/system/error_code.hpp>
#include "plymeshexporter.hpp"

TDV_NAMESPACE_BEGIN

void PLYMeshExporter::exportMesh(const GridGLMesh &mesh)    
{
    std::ofstream outstream(m_filename.c_str());
    
    if ( !outstream.good() )
    {
        boost::system::error_code errcode;
        throw Exception(boost::format("Can't open file %1%: %2%") 
                        % m_filename % errcode.message());
    }

    const Dim dim(mesh.dim());
    const size_t nTrigs = (dim.width() - 1)*(dim.height() - 1);
        
    using namespace std;
    
    outstream << "ply" << endl
              << "format ascii 1.0" << endl
              << "comment VCGLIB generated" << endl
              << "element vertex " << dim.size() << endl
              << "property float x" << endl
              << "property float y" << endl
              << "property float z" << endl
              << "property uchar red" << endl       
              << "property uchar green" << endl
              << "property uchar blue" << endl
              << "element face " << nTrigs << endl
              << "property list uchar int vertex_indices" << endl
              << "end_header" << endl;    

    ud::Vec3f vert, color;    
    for (size_t row=0; row<dim.height(); row++)
    {
        for (size_t col=0; col<dim.width(); col++)
        {
            mesh.point(col, row, vert, color);
            
            outstream << vert[0] << ' ' << vert[1] << ' ' << vert[2] << ' '
                      << static_cast<int>(color[0]*255.0f) << ' ' 
                      << static_cast<int>(color[1]*255.0f) << ' ' 
                      << static_cast<int>(color[2]*255.0f) << ' ' << endl;
        }
    }
    
    for (size_t row=0; row<dim.height() - 1; row++)
    {
        for (size_t col=0; col<dim.width() - 1; col++)
        {
            /**
             * a_d
             * | |
             * b-c
             */
            const size_t a = row*dim.width() + col;
            const size_t b = (row + 1)*dim.width() + col;
            const size_t c = (row + 1)*dim.width() + col + 1;             
            const size_t d = row*dim.width() + col + 1;
            
            assert( a < dim.size());
            assert( b < dim.size());
            assert( c < dim.size());
            
            outstream << "4 " << a << ' ' << b << ' ' 
                      << c << ' ' << d << ' '<< endl;
        }
    }

    outstream.close();
}

TDV_NAMESPACE_END
