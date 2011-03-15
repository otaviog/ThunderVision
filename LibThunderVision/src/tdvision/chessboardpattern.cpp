#include "chessboardpattern.hpp"

TDV_NAMESPACE_BEGIN

ChessboardPattern::ChessboardPattern()
{
    m_dim = cvSize(8, 8);    
    m_squareSize = 1.0f;        
}

void ChessboardPattern::generateObjectPoints(std::vector<CvPoint3D32f> &objPts) const
{
    for (size_t i=0; i<m_dim.height; i++)
    {
        for (size_t j=0; j<m_dim.width; j++)
        {
            objPts[i*m_dim.width + j] = 
                cvPoint3D32f(i*m_squareSize, j*m_squareSize, 0);
        }
    }
}

TDV_NAMESPACE_END
