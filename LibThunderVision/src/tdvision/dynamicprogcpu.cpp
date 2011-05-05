#include <boost/scoped_array.hpp>
#include "cuerr.hpp"
#include "dynamicprogcpu.hpp"

TDV_NAMESPACE_BEGIN

inline size_t dsiOffset(
    const Dim &dim, size_t x, size_t y, size_t z)
{
    return z + dim.depth()*y + dim.height()*dim.depth()*x;
}

static void dynamicProg(const Dim &dim, const float *dsi, size_t y, float *cost, char *path)
{
    for (size_t z=0; z<dim.depth(); z++)
    {
        const size_t offset = dsiOffset(dim, 0, y, z);
        cost[offset] = dsi[offset];
        path[offset] = 0;
    }
    
    for (size_t x=1; x<dim.width(); x++)
    {
        for (size_t z=0; z<dim.depth(); z++)
        {
            const size_t offset = dsiOffset(dim, x, y, z);
            const float c0 = dsi[offset];
            
            const float c1 = (z > 0) 
                ? cost[dsiOffset(dim, x - 1, y, z - 1)]
                : std::numeric_limits<float>::infinity();
            
            const float c2 = cost[dsiOffset(dim, x - 1, y, z)];
            
            const float c3 = (z < dim.depth() - 1)
                ? cost[dsiOffset(dim, x - 1, y, z + 1)]
                : std::numeric_limits<float>::infinity();
            
            float m;
            char p;
            
            if ( c1 < c2 && c1 < c3 )
            {
                m = c1;
                p = -1;
            }
            else if ( c2 < c3 )
            {
                m = c2;
                p = 0;
            }
            else
            {
                m = c3;
                p = 1;
            }
                        
            cost[offset] = c0 + m;            
            path[offset] = p;
        }
    }
}

static void reducePath(const Dim &dim, size_t y, const float *cost, 
                       const char *path, float *image)
{
    float minZValue = std::numeric_limits<float>::infinity();
    size_t minZ = 0;
    for (size_t z=0; z<dim.depth(); z++)
    {
        const size_t offset = dsiOffset(dim, dim.width() - 1, y, z);
        const float value = cost[offset];
        
        if ( value < minZ )
        {
            minZValue = value;
            minZ = z;
        }
    }
    
    for (int x=static_cast<int>(dim.width() - 1); x >= 0; x--)
    {        
        const size_t p = minZ + path[dsiOffset(dim, x, y, minZ)];
        
        if ( p >= 0 && p < dim.depth() )
        {
            minZ = p;
        }
        
        image[dim.width()*y + x] = float(minZ)/float(dim.depth());
    }
}

void DynamicProgCPU::updateImpl(DSIMem mem, FloatImage img)
{
    CUerrExp cuerr;
    const Dim &dim = mem.dim();
    boost::scoped_array<float> dsi(new float[dim.size()]);
    
    cuerr << cudaMemcpy(dsi.get(), mem.mem(), sizeof(float)*dim.size(),
                        cudaMemcpyDeviceToHost);
    
    boost::scoped_array<float> cost(new float[dim.size()]);
    boost::scoped_array<char> path(new char[dim.size()]);    
    
    for (size_t y=0; y<dim.height(); y++)
    {
        dynamicProg(dim, dsi.get(), y, cost.get(), path.get());
        reducePath(dim, y, cost.get(), path.get(), img.cpuMem()->data.fl);
    }
}

TDV_NAMESPACE_END
