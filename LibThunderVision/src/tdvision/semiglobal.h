#ifndef TDV_SEMIGLOBAL_H
#define TDV_SEMIGLOBAL_H

#include <tdvbasic/common.hpp>
#include "dim.hpp"

TDV_NAMESPACE_BEGIN

#define SG_MAX_PASSES 4
#define SG_MAX_PATHS 2048

struct SGPoint
{
    int x, y;
};

struct SGPathsDesc
{      
    int maxPathsSizes[SG_MAX_PATHS];
    int pathsSizes[SG_MAX_PASSES][SG_MAX_PATHS];
    SGPoint pathsStarts[SG_MAX_PASSES][SG_MAX_PATHS];
    SGPoint pathsEnds[SG_MAX_PASSES][SG_MAX_PATHS];
};

class SGPaths
{
public:
    SGPaths();
  
    SGPathsDesc* getDesc(const Dim &imgDim);
  
    void unalloc();
  
private: 
    void horizontalPathsDesc(uint totalPaths, int *pathSizes, 
                             SGPoint *pathsStarts, SGPoint *pathsEnds);

    void verticalPathsDesc(uint totalPaths, int *pathSizes, 
                           SGPoint *pathsStarts, SGPoint *pathsEnds);

    void topBottomDiagonaDesc(int *pathSizes, SGPoint *pathsStarts, 
                              SGPoint *pathsEnds);
  
    void bottomTopDiagonalDesc(int *pathSizes, SGPoint *pathsStarts, 
                               SGPoint *pathsEnds);

    SGPathsDesc *m_desc_d;
    tdv::Dim m_imgDim;
};

TDV_NAMESPACE_END

#endif /* TDV_SEMIGLOBAL_H */
