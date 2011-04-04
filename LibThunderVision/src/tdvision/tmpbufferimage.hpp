#ifndef TDV_TMPBUFFERIMAGE_HPP
#define TDV_TMPBUFFERIMAGE_HPP

#include <tdvbasic/common.hpp>
#include <cv.h>

TDV_NAMESPACE_BEGIN

class TmpBufferImage
{
public:
    TmpBufferImage(int type);
    
    ~TmpBufferImage();
    
    TmpBufferImage(const TmpBufferImage &cpy)
    {
        copy(cpy);
    }
    
    TmpBufferImage& operator=(const TmpBufferImage &cpy)
    {
        if ( m_img != NULL )
        {
            cvReleaseMat(&m_img);
            m_img = NULL;
        }
        
        copy(cpy);
        return *this;
    }
    
    CvMat* getImage(int width, int height);
    
    CvMat* getImage(const CvSize &sz)
    {
        return getImage(sz.width, sz.height);
    }
    
private:
    void copy(const TmpBufferImage &cpy);
    
    CvMat *m_img;
    int m_type;
};

TDV_NAMESPACE_END

#endif /* TDV_TMPBUFFERIMAGE_HPP */
