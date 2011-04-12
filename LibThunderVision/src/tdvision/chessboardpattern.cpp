#include "chessboardpattern.hpp"

TDV_NAMESPACE_BEGIN

ChessboardPattern::ChessboardPattern(const Dim &dim)
{
    //m_dim = cvSize(7, 6);
    m_dim = cvSize(dim.width(), dim.height());
    m_squareSize = 1.0f;        
}

void ChessboardPattern::generateObjectPoints(std::vector<CvPoint3D32f> &objPts) const
{
    for (int i=0; i<m_dim.height; i++)
    {
        for (int j=0; j<m_dim.width; j++)
        {
            objPts[i*m_dim.width + j] = 
                cvPoint3D32f(i*m_squareSize, j*m_squareSize, 0);
        }
    }
}

TDV_NAMESPACE_END
