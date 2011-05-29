#ifndef TDV_MISC_HPP
#define TDV_MISC_HPP

#include <tdvbasic/common.hpp>
#include <cv.h>

TDV_NAMESPACE_BEGIN

namespace misc
{
    CvMat* create8UGray(const CvArr *src); 
    
    CvMat *create32FGray(const CvArr *src);
    
    void convert8UC3To32FC1Gray(const CvArr *src, CvArr *dst, 
                                CvArr *tmpGray = NULL);        
    
    
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
