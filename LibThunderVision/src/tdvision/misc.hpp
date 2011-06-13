#ifndef TDV_MISC_HPP
#define TDV_MISC_HPP

#include <tdvbasic/common.hpp>
#include <cv.h>
#include "tmpbufferimage.hpp"

TDV_NAMESPACE_BEGIN

namespace misc
{
    CvMat* create8UGray(const CvArr *src); 
    
    CvMat *create32FGray(const CvArr *src);
    
    void convert8UC3To32FC1Gray(const CvArr *src, CvArr *dst, CvArr *tmpRGBF = NULL);
    
    void convert8UC3To32FC1GrayHSV(const CvArr *src, CvArr *dst, CvArr *hsvTmp, CvArr *u8Tmp);
    
    class Conv8UC3To32FC1
    {
    public:
        Conv8UC3To32FC1()
            : m_rgb32f(CV_32FC3)
        {
            
        }

        void convert(const CvArr *src, CvArr *dst)
        {
            convert8UC3To32FC1Gray(src, dst, 
                                   m_rgb32f.getImage(cvGetSize(src))); 
        }
        
    private:
        TmpBufferImage m_rgb32f;
    };
        
    class Conv8UC3To32FC1Hsv
    {
    public:
        Conv8UC3To32FC1Hsv()
            : m_hsvTmp(CV_8UC3), m_8uTmp(CV_8UC1)
        {
        }
        
        void convert(const CvArr *src, CvArr *dst)
        {
            const CvSize size(cvGetSize(src));
            convert8UC3To32FC1GrayHSV(src, dst, 
                                      m_hsvTmp.getImage(size), 
                                      m_8uTmp.getImage(size));
        }
        
    private:
        TmpBufferImage m_hsvTmp, m_8uTmp;
    };

    class ScopedMat
    {
    public:
        ScopedMat(CvMat *mat)
        {
            m_mat = mat;
        }
        
        ~ScopedMat()
        {
            cvReleaseMat(&m_mat);
        }
        
        CvMat* get() 
        {
            return m_mat;
        }
        
        const CvMat* get() const
        {
            return m_mat;
        }
        
    private:
        CvMat *m_mat;
    };        
}

TDV_NAMESPACE_END

#endif /* TDV_MISC_HPP */
