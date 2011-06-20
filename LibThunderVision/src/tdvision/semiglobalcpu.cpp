#include <boost/scoped_array.hpp>
#include <cassert>
#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/blocked_range.h>
#include "cuerr.hpp"
#include "semiglobalcpu.hpp"
#include "semiglobal.h"
#include "wtacpu.hpp"
#include <highgui.h>

#include "benchmark.hpp"
#include <iostream>
#include <cstdio>

TDV_NAMESPACE_BEGIN

inline size_t dsiOffset(
    const Dim &dim, size_t x, size_t y, size_t z)
{
    return z + dim.depth()*y + dim.height()*dim.depth()*x;
}

inline float dsiValue(const Dim &dim, const float *mem, size_t x, size_t y, 
                      size_t z)
{
    const size_t offset = dsiOffset(dim, x, y, z);
    if ( offset < dim.size() )
    {
        return mem[offset];
    }

    return 0.0f;
}

const float P1 = 30.0f/255.0f;
const float P2 = 150.0f/255.0f;

#define min4(a, b, c, d) std::min(a, std::min(b, std::min(c, d)))

static void zeroVolume(const Dim &dim,
                       float *vol)
{
    for (size_t y=0; y<dim.height(); y++)
    {
        for (size_t x=0; x<dim.width(); x++)
        {
            for (size_t d=0; d<dim.depth(); d++)
            {
                vol[dsiOffset(dim, y, x, d)] = 0.0f;
            }
        }
    }
}

void costPath(const Dim &dsiDim,
              const float *dsi,
              const float *img,
              const SGPoint &start,
              const sSGPoint &dir,
              size_t pathLength,
              float *aggregVol,
              float *lastCost,
              float *newCost)
{
    
    float lastIntensity;
    SGPoint pt = start;
    
    //printf("%d\n", dsi);
#if 1
    for (size_t z=0; z<dsiDim.depth(); z++)
    {
        const int X = pt.x;
        const int Y = pt.y;
        const int dz = z + 1;
        
        const size_t dsiOff = dsiOffset(dsiDim, X, Y, z);
        assert(dsiOff < dsiDim.size());        
        const float cost = dsi[dsiOff];
        
        aggregVol[dsiOff] = cost;
        lastCost[dz] = cost;
        lastIntensity = img[Y*dsiDim.width() + X];        
    }
#endif

    for (size_t x=1; x<pathLength; x++)
    {        
        pt.x += dir.x;
        pt.y += dir.y;

        const int X = pt.x;
        const int Y = pt.y;

        float minCost = lastCost[0];
        for (size_t z=1; z<dsiDim.depth(); z++)
            minCost = min(lastCost[z], minCost);

        const float intensity = img[Y*dsiDim.width() + X];
        const float P2Adjust = P2/std::abs(intensity - lastIntensity);

        lastIntensity = intensity;

        for (size_t z=0; z<dsiDim.depth(); z++)
        {
            const size_t dsiOff =
                dsiOffset(dsiDim, X, Y, z);
            
            assert(dsiOff < dsiDim.size());
            
            const float cost = dsi[dsiOff];

            const int dz = z + 1;
            
            const float Lr =
                cost + min4(lastCost[dz],
                            lastCost[dz - 1] + P1,
                            lastCost[dz + 1] + P1,
                            minCost + P2Adjust) - minCost;
            aggregVol[dsiOff] += Lr;
            newCost[dz] = Lr;            
        }

        std::swap(newCost, lastCost);
    }
}

struct PathCostParallel
{
public:
    PathCostParallel(const Dim &d, const float * const ds,
             const float * const i, const SGPath * const p,
             float * const a)
        : dim(d), dsi(ds), img(i), paths(p),
          aggregVol(a)
    { }

    void operator()(const tbb::blocked_range<size_t> &br) const
    {
        const size_t depth = dim.depth();
        
        boost::scoped_array<float> lastCost(new float[depth + 2]);
        boost::scoped_array<float> newCost(new float[depth + 2]);
        
        lastCost[0] = std::numeric_limits<float>::infinity();
        newCost[0] = std::numeric_limits<float>::infinity();        
        lastCost[depth + 1] = std::numeric_limits<float>::infinity();
        newCost[depth + 1] = std::numeric_limits<float>::infinity();

        for (size_t p=br.begin(); p != br.end(); p++)
        {
            const SGPath &ph = paths[p];
            costPath(dim, dsi, img, ph.start, 
                     ph.dir, ph.size, aggregVol, lastCost.get(),
                     newCost.get());

            sSGPoint invDir = {-ph.dir.x, -ph.dir.y};
            
            costPath(dim, dsi, img, ph.end, invDir, ph.size,
                     aggregVol, lastCost.get(),
                     newCost.get());
        }
    }

private:
    const Dim &dim;
    const float * const dsi;
    const float * const img;
    const SGPath * const paths;
    float * const aggregVol;
};

void SemiGlobalCPU::updateImpl(DSIMem mem, FloatImage img)
{
    const Dim &dim = mem.dim();
    const Dim &imgDim = img.dim();

    float *dsi = (float*) mem.toCpuMem();
    printf("Z %d\n", dsi);
    boost::scoped_array<float> aggreg(new float[dim.size()]);
    printf("A %d\n", dsi);
    
    size_t pathCount;
    boost::scoped_array<SGPath> paths(
        SGPaths::getDescCPU(imgDim, &pathCount));
    
    printf("B %d\n", dsi);
    //zeroVolume(dim, aggreg.get());
    //tbb::task_scheduler_init init;

    CudaBenchmarker bMarker;       
    bMarker.begin();
#if 0
    tbb::parallel_for(tbb::blocked_range<size_t>(0, pathCount),
                      PathCostParallel(dim, dsi, img.cpuMem()->data.fl,
                                       paths.get(), aggreg.get()),
                      tbb::auto_partitioner());
#elif 1

    const size_t depth = dim.depth();
    
    const float *imgMem = img.cpuMem()->data.fl;
    
    boost::scoped_array<float> lastCost(new float[depth + 2]);
    boost::scoped_array<float> newCost(new float[depth + 2]);
        
    lastCost[0] = std::numeric_limits<float>::infinity();
    newCost[0] = std::numeric_limits<float>::infinity();        
    lastCost[depth + 1] = std::numeric_limits<float>::infinity();
    newCost[depth + 1] = std::numeric_limits<float>::infinity();

    for (size_t p=0; p < pathCount; p++)
    {
        const SGPath &ph = paths[p];
        costPath(dim, dsi, imgMem, ph.start, 
                 ph.dir, ph.size, aggreg.get(), lastCost.get(),
                 newCost.get());

        sSGPoint invDir = {-ph.dir.x, -ph.dir.y};
            
        costPath(dim, dsi, imgMem, ph.end, 
                 invDir, ph.size, aggreg.get(), lastCost.get(),
                 newCost.get());
    }
#endif
        
    WTACPU::wta(dim, aggreg.get(), img.cpuMem()->data.fl);
    
    bMarker.end();
    Benchmark bmark = bMarker.elapsedTime();
    std::cout << bmark.secs() << std::endl;    
}

TDV_NAMESPACE_END
