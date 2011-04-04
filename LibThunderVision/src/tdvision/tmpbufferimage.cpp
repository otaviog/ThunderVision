#include "tmpbufferimage.hpp"

TDV_NAMESPACE_BEGIN

TmpBufferImage::TmpBufferImage(int type)
{
    m_img = NULL;
    m_type = type;
}

TmpBufferImage::~TmpBufferImage()
{
    if ( m_img != NULL )
    {
        cvReleaseMat(&m_img);
        m_img = NULL;
    }
}

CvMat* TmpBufferImage::getImage(int width, int height)
{
    if ( m_img == NULL || m_img->cols != width || m_img->rows != height )
    {
        m_img = cvCreateMat(height, width, m_type);
    }    
        
    return m_img;
}

void TmpBufferImage::copy(const TmpBufferImage &cpy)
{    
    if ( cpy.m_img == NULL )
    {
        m_img = NULL;        
    }
    else
    {
        m_img = cvCloneMat(cpy.m_img);
    }

    m_type = cpy.m_type;
}

TDV_NAMESPACE_END
