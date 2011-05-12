#ifndef TDV_REPROJECTIONPROCESS_HPP
#define TDV_REPROJECTIONPROCESS_HPP

#include <tdvbasic/common.hpp>
#include <cv.h>
#include "floatimage.hpp"
#include "process.hpp"
#include "reprojection.hpp"

TDV_NAMESPACE_BEGIN

class ReprojectProcess: public Process
{
public:
    ReprojectProcess(Reprojection *reproj = NULL, Reprojector *proj = NULL)
    {
        m_dispPipe = NULL;
        m_originPipe = NULL;
        m_reproj = reproj;
        m_proj = proj;
    }
    
    void setReprojection(Reprojection *reproj)
    {
        m_reproj = reproj;
    }
    
    void setReprojector(Reprojector *proj)
    {
        m_proj = proj;
    }

    void input(ReadPipe<FloatImage> *dispPipe, ReadPipe<CvMat*> *originPipe)
    {
        m_dispPipe = dispPipe;
        m_originPipe = originPipe;
    }
    
    void process();
    
private:
    ReadPipe<FloatImage> *m_dispPipe;
    ReadPipe<CvMat*> *m_originPipe;
    Reprojection *m_reproj;
    Reprojector *m_proj;
};
    
TDV_NAMESPACE_END

#endif /* TDV_REPROJECTIONPROCESS_HPP */
