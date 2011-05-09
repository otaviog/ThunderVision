#ifndef TDV_CHESSBOARDPATTERN_HPP
#define TDV_CHESSBOARDPATTERN_HPP

#include <tdvbasic/common.hpp>
#include <cv.h>
#include "dim.hpp"

TDV_NAMESPACE_BEGIN

class ChessboardPattern
{
public:
    ChessboardPattern(const Dim &dim = Dim(9, 6));
        
    CvSize dim()
    {
        return m_dim;
    }
    
    size_t totalCorners() const
    {
        return m_dim.width*m_dim.height;
    }
    
    void generateObjectPoints(std::vector<CvPoint3D32f> &v) const;
    
private:       
    CvSize m_dim;    
    float m_squareSize;
};


TDV_NAMESPACE_END

#endif /* TDV_CHESSBOARDPATTERN_HPP */
