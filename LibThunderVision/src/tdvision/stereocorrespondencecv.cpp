#include "sink.hpp"
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
#if 0 
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
        
        FloatImageSinkPol::sink(limg);
        FloatImageSinkPol::sink(rimg);
    }
    
    return wguard.wasWrite();
#else
    FloatImage limg, rimg;
    if ( m_lrpipe->read(&limg) && m_rrpipe->read(&rimg) )
    {
        FloatImageSinkPol::sink(limg);
        FloatImageSinkPol::sink(rimg);

        return true;
    }
    
    return false;
#endif
}

TDV_NAMESPACE_END
