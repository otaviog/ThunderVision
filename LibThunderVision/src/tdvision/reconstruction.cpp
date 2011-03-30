#include "stereomatcher.hpp"
#include "reconstruction.hpp"

TDV_NAMESPACE_BEGIN

Reconstruction::Reconstruction(StereoMatcher *matcher, 
                               ReadPipe<IplImage*> *leftImgIn,
                               ReadPipe<IplImage*> *rightImgIn)
{
    m_matcher = matcher;
    
    m_ctrlProc.inputs(leftImgIn, rightImgIn);
    
    m_rectify.leftImgInput(m_ctrlProc.leftImgOutput());    
    m_rectify.rightImgInput(m_ctrlProc.rightImgOutput());    
    
    m_rectTee[0].input(m_rectify.leftImgOutput());
    m_rectTee[1].input(m_rectify.rightImgOutput());
                
    m_rectTee[0].enable(0);
    m_rectTee[1].enable(0);
    
    m_matcher->inputs(m_rectTee[0].output(0),
                      m_rectTee[1].output(0));
    
    m_procs.addProcess(&m_ctrlProc);
    m_procs.addProcess(&m_rectify);
    m_procs.addProcess(&m_rectTee[0]);
    m_procs.addProcess(&m_rectTee[1]);
    m_procs.addProcess(*m_matcher);    
}

void Reconstruction::dupRectficatin(ReadPipe<FloatImage> **leftRectOut, 
                                    ReadPipe<FloatImage> **rightRectOut)
{
    m_rectTee[0].enable(1);
    m_rectTee[1].enable(1);
    
    *leftRectOut = m_rectTee[0].output(1);
    *rightRectOut = m_rectTee[1].output(1);        
}

void Reconstruction::undupRectification()
{
    m_rectTee[0].disable(1);
    m_rectTee[1].disable(1);
}

TDV_NAMESPACE_END
