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

class Reconstruction: private ArrayProcessGroup
{
public:
    Reconstruction(StereoMatcher *matcher, 
                   ReadPipe<IplImage*> *leftImgIn,
                   ReadPipe<IplImage*> *rightImgIn);
    
    Process** processes()
    {
        return ArrayProcessGroup::processes();
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
    
    void dupRectficatin(ReadPipe<FloatImage> **leftRectOut, 
                        ReadPipe<FloatImage> **rightRectOut);

    void undupRectification();
    
private:
    CtrlProcess m_ctrlProc;
    TWorkUnitProcess<RectificationCV> m_rectify;
    TWorkUnitProcess<TeeWorkUnit<FloatImage> > m_rectTee[2];    
    StereoMatcher *m_matcher;    
};


TDV_NAMESPACE_END

#endif /* TDV_RECONSTRUCTION_HPP */
