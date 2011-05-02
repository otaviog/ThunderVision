#ifndef TDV_RECONSTRUCTION_HPP
#define TDV_RECONSTRUCTION_HPP

#include <tdvbasic/common.hpp>
#include "processgroup.hpp"
#include "ctrlprocess.hpp"
#include "rectificationcv.hpp"
#include "floatconv.hpp"
#include "teeworkunit.hpp"
#include "workunitprocess.hpp"

TDV_NAMESPACE_BEGIN

class StereoMatcher;
class Benchmark;

class Reconstruction: public ProcessGroup
{
public:
    Reconstruction(StereoMatcher *matcher, 
                   ReadPipe<CvMat*> *leftImgIn,
                   ReadPipe<CvMat*> *rightImgIn);
    
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
    }
    
    void pause()
    {
        m_ctrlProc.pause();
    }
    
    const Benchmark& benchmark() const;
    
    void dupRectification(ReadPipe<FloatImage> **leftRectOut, 
                          ReadPipe<FloatImage> **rightRectOut);

    void undupRectification();
    
    void dupDisparityMap(ReadPipe<FloatImage> **dispMapOut);
    
    void undupDisparityMap();
    
    void camerasDesc(const CamerasDesc &desc)
    {
        m_rectify.camerasDesc(desc);
    }
    
private:
    CtrlProcess m_ctrlProc;
    TWorkUnitProcess<RectificationCV> m_rectify;
    TWorkUnitProcess<TeeWorkUnit<FloatImage> > m_rectTee[2];    
    TWorkUnitProcess<TeeWorkUnit<FloatImage> > m_dispTee;    
    StereoMatcher *m_matcher;    
    
    ArrayProcessGroup m_procs;
};


TDV_NAMESPACE_END

#endif /* TDV_RECONSTRUCTION_HPP */
