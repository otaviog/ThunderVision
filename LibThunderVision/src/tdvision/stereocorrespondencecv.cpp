#include <highgui.h>
#include "sink.hpp"
#include "stereocorrespondencecv.hpp"

TDV_NAMESPACE_BEGIN

StereoCorrespondenceCV::StereoCorrespondenceCV(
    MatchingMode mode, int maxDisparity, int maxIterations)
    : m_limg8U(CV_8U), m_rimg8U(CV_8U)
{
    if ( mode == LocalMatching )
    {
        m_bmState = cvCreateStereoBMState(CV_STEREO_BM_BASIC, maxDisparity);
        workName("OpenCV_BlockMatcher");
    }
    else
    {
        m_gcState = cvCreateStereoGCState(maxDisparity, maxIterations);
        workName("OpenCV_GraphCut");
    }

    m_mode = mode;
        
}

StereoCorrespondenceCV::~StereoCorrespondenceCV()
{
    if ( m_mode == LocalMatching )
        cvReleaseStereoBMState(&m_bmState);
    else
        cvReleaseStereoGCState(&m_gcState);
}

bool StereoCorrespondenceCV::update()
{
#if 1
    WriteGuard<ReadWritePipe<FloatImage> > wguard(m_wpipe);
    FloatImage limg, rimg;

    if ( m_lrpipe->read(&limg) && m_rrpipe->read(&rimg) )
    {
        CvMat *limg_c = limg.cpuMem();
        CvMat *rimg_c = rimg.cpuMem();
        
        CvMat *limg8u_c = m_limg8U.getImage(cvGetSize(limg_c));
        CvMat *rimg8u_c = m_rimg8U.getImage(cvGetSize(rimg_c));

        cvConvertScale(limg_c, limg8u_c, 255.0);
        cvConvertScale(rimg_c, rimg8u_c, 255.0);

        FloatImage output = FloatImage::CreateCPU(
            Dim::minDim(limg.dim(), rimg.dim()));
        CvMat *out_c = output.cpuMem();

        if ( m_mode == LocalMatching )
            cvFindStereoCorrespondenceBM(limg8u_c, rimg8u_c, out_c, m_bmState);
        else
            cvFindStereoCorrespondenceGC(limg8u_c, rimg8u_c, out_c, NULL,
                                         m_gcState);

        wguard.write(output);

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
