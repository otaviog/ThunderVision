#include <limits>
#include "stereomatcher.hpp"
#include "reconstruction.hpp"

TDV_NAMESPACE_BEGIN

/**
 * leftImgIn-|
 *           |-leftOriginTee
 *
 *
 * rightImgIn-|
 *
 */

//#define TDV_RESIZE_STEREO 

Reconstruction::Reconstruction(StereoMatcher *matcher,
                               ReadPipe<CvMat*> *leftImgIn,
                               ReadPipe<CvMat*> *rightImgIn,
                               Reprojection *reprojection)
    : m_dispTee(this, &m_bcallback)
{
    m_bcallback = NULL;
    m_matcher = matcher;

    m_ctrlProc.inputs(leftImgIn, rightImgIn);

    m_rectify.leftImgInput(m_ctrlProc.leftImgOutput());
    m_rectify.rightImgInput(m_ctrlProc.rightImgOutput());

    m_rectTee[0].input(m_rectify.leftImgOutput());
    m_rectTee[1].input(m_rectify.rightImgOutput());

    m_rectTee[0].enable(RECT_MATCHER_ID);
    m_rectTee[1].enable(RECT_MATCHER_ID);

#ifdef TDV_RESIZE_STEREO
    m_resize[0].input(m_rectTee[0].output(RECT_MATCHER_ID));
    m_resize[1].input(m_rectTee[1].output(RECT_MATCHER_ID));
    
    m_matcher->inputs(m_resize[0].output(), m_resize[1].output());
#else
    m_matcher->inputs(m_rectTee[0].output(RECT_MATCHER_ID), m_rectTee[1].output(RECT_MATCHER_ID));
#endif
    
    m_dispTee.input(m_matcher->output());
    m_dispTee.enable(DISP_REP_ID);

    m_reprojectProc.input(m_dispTee.output(DISP_REP_ID),
                          m_rectify.colorImgOutput());

    if ( reprojection != NULL )
    {
        m_reprojectProc.setReprojection(reprojection);
        m_reprojectProc.setReprojector(&m_rectify);
        //m_reprojectProc.setReprojector(&m_altReproj);
        m_rectify.enableColorRemap();
    }

    m_procs.addProcess(&m_ctrlProc);
    m_procs.addProcess(&m_rectify);
    m_procs.addProcess(&m_rectTee[0]);
    m_procs.addProcess(&m_rectTee[1]);
#ifdef TDV_RESIZE_STEREO
    m_procs.addProcess(&m_resize[0]);
    m_procs.addProcess(&m_resize[1]);
#endif
    m_procs.addProcess(*m_matcher);
    m_procs.addProcess(&m_dispTee);
    m_procs.addProcess(&m_reprojectProc);
    
    m_ctrlProc.workName("Ctrl Process");
    m_rectify.workName("Rectification");
    m_rectTee[0].workName("Rectification Tee 0");
    m_rectTee[1].workName("Rectification Tee 1");
    m_dispTee.workName("Disparity Tee");
}

void Reconstruction::dupRectification(
    ReadPipe<FloatImage> **leftRectOut,
    ReadPipe<FloatImage> **rightRectOut)
{
    m_rectTee[0].enable(RECT_VIEW_ID);
    m_rectTee[1].enable(RECT_VIEW_ID);

    *leftRectOut = m_rectTee[0].output(RECT_VIEW_ID);
    *rightRectOut = m_rectTee[1].output(RECT_VIEW_ID);
}

void Reconstruction::undupRectification()
{
    m_rectTee[0].disable(RECT_VIEW_ID);
    m_rectTee[1].disable(RECT_VIEW_ID);
}

void Reconstruction::dupDisparityMap(ReadPipe<FloatImage> **dispMapOut)
{
    m_dispTee.enable(DISP_VIEW_ID);
    *dispMapOut = m_dispTee.output(DISP_VIEW_ID);
}

void Reconstruction::undupDisparityMap()
{
    m_dispTee.disable(DISP_VIEW_ID);
}

void Reconstruction::DispTeeProcess::process()
{
    bool cont = update();

    while ( cont )
    {
        cont = update();

        if ( *m_callback != NULL )
        {
            const float pbs = packetsBySeconds();
            if ( m_self->mode() == FlowCtrl::Step )
            {
                (*m_callback)->reconstructionDone(30);
            }
            else if ( pbs < std::numeric_limits<float>::infinity() )
                (*m_callback)->reconstructionDone(pbs);
        }
    }
}

TDV_NAMESPACE_END
