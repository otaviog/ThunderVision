#include "stereocorrespondencecv.hpp"

TDV_NAMESPACE_BEGIN

StereoCorrespondenceCV::StereoCorrespondenceCV()
{
    m_bmState = cvCreateStereoBMState(CV_STEREO_BM_BASIC, 255);
}

StereoCorrespondenceCV::~StereoCorrespondenceCV()
{
    cvReleaseStereoBMState(&m_bmState);
}

bool StereoCorrespondenceCV::update()
{
    WriteGuard<ReadWritePipe<FloatImage> > wguard(m_wpipe);
    FloatImage limg, rimg;
    
    if ( m_lrpipe->read(&limg) && m_rrpipe->read(&rimg) )
    {
        CvMat *limg_c = limg.cpuMem();
        CvMat *rimg_c = rimg.cpuMem();
        
        FloatImage output = FloatImage::CreateCPU(
            Dim::minDim(limg.dim(), rimg.dim()));
        CvMat *out_c = output.cpuMem();
        
        cvFindStereoCorrespondenceBM(limg_c, rimg_c, out_c, m_bmState);
    }
    
    return wguard.wasWrite();
}

TDV_NAMESPACE_END
