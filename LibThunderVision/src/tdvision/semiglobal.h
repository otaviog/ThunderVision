#ifndef TDV_SEMIGLOBAL_H
#define TDV_SEMIGLOBAL_H

#include <tdvbasic/common.hpp>
#include "dim.hpp"

TDV_NAMESPACE_BEGIN

struct SGPoint
{
    unsigned short x, y;
};

struct sSGPoint
{
    short x, y;
};

struct SGPath
{
    SGPoint start, end;
    sSGPoint dir;
    unsigned short size;    
};

class SGPaths
{
public:
    SGPaths();
  
    SGPath* getDescDev(const Dim &imgDim);
    
    static SGPath* getDescCPU(const Dim &imgDim, size_t *pathCount);
    
    size_t pathCount() const
    {
        return m_pathCount;
    }
    
    void unalloc();
  
private: 
    static void horizontalPathsDesc(const Dim &imgDim, SGPath *paths);

    static void verticalPathsDesc(const Dim &imgDim, SGPath *paths);

    static void topBottomDiagonaDesc(const Dim &imgDim, SGPath *paths);
    
    static void bottomTopDiagonalDesc(const Dim &imgDim, SGPath *paths);

    SGPath *m_paths_d;
    tdv::Dim m_imgDim;
    size_t m_pathCount;
};

void SemiGlobalRunDev(const tdv::Dim &dsiDim, cudaPitchedPtr dsi,
                      const tdv::SGPath *pathsArray, size_t pathCount,
                      const float *lorigin, cudaPitchedPtr aggregDSI, 
                      float *dispImg, bool zeroAggregDSI);

TDV_NAMESPACE_END

#endif /* TDV_SEMIGLOBAL_H */
