#ifndef TDV_RECONSTRUCTION_HPP
#define TDV_RECONSTRUCTION_HPP

#include <tdvbasic/common.hpp>
#include "processgroup.hpp"
#include "ctrlprocess.hpp"
#include "rectificationcv.hpp"
#include "imageresize.hpp"
#include "teeworkunit.hpp"
#include "workunitprocess.hpp"
#include "reprojectprocess.hpp"
#include "dilate.hpp"
#include "directreprojector.hpp"

TDV_NAMESPACE_BEGIN

class StereoMatcher;
class Benchmark;

class Reconstruction: public ProcessGroup
{
public:
    class BenchmarkCallback
    {
    public:
        virtual void reconstructionDone(float time) = 0;
    };

private:
    class DispTeeProcess: public Process, public TeeWorkUnit<FloatImage>
    {
    public:
        DispTeeProcess(Reconstruction *self, BenchmarkCallback **callback)
        {
            m_callback = callback;
            m_self = self;
        }

        void process();

    private:
        Reconstruction *m_self;
        BenchmarkCallback **m_callback;
    };

    static const int LOGN_RECT_ID = 0;
    static const int LOGN_REP_ID = 1;
    static const int DISP_REP_ID = 0;
    static const int DISP_VIEW_ID = 1;   
    static const int RECT_REPROJ_L_ID = 0;
    static const int RECT_VIEW_ID = 1;

public:
    Reconstruction(StereoMatcher *matcher,
                   ReadPipe<CvMat*> *leftImgIn,
                   ReadPipe<CvMat*> *rightImgIn,
                   Reprojection *reprojection);

    Process** processes()
    {
        return m_procs.processes();
    }

    void continuous()
    {
        m_ctrlProc.continuous();
    }

    void step()
    {
        m_ctrlProc.step();
        if ( m_bcallback != NULL )
            m_bcallback->reconstructionDone(30);
    }

    void pause()
    {
        m_ctrlProc.pause();
    }

    FlowCtrl::Mode mode() const
    {
        return m_ctrlProc.mode();
    }

    const Benchmark& benchmark() const;

    void dupRectification(ReadPipe<CvMat*> **leftRectOut,
                          ReadPipe<CvMat*> **rightRectOut);

    void undupRectification();

    void dupDisparityMap(ReadPipe<FloatImage> **dispMapOut);

    void undupDisparityMap();

    void camerasDesc(const CamerasDesc &desc)
    {
        m_rectify.camerasDesc(desc);
    }

    void benchmarkCallback(BenchmarkCallback *callback)
    {
        m_bcallback = callback;
    }

private:
    CtrlProcess m_ctrlProc;
    TWorkUnitProcess<RectificationCV> m_rectify;    
    TWorkUnitProcess<TeeWorkUnit<CvMat*> > m_rectTee[2];
    TWorkUnitProcess<ImageResize> m_resize[2];
    StereoMatcher *m_matcher;
    TWorkUnitProcess<Dilate> m_dilate;
    DispTeeProcess m_dispTee;
    ReprojectProcess m_reprojectProc;

    ArrayProcessGroup m_procs;

    BenchmarkCallback *m_bcallback;
    
    
    DirectReprojector m_altReproj;
};


TDV_NAMESPACE_END

#endif /* TDV_RECONSTRUCTION_HPP */
