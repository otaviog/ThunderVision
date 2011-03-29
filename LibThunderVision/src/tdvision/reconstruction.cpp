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
                
    m_matcher->inputs(m_rectTee[0].output1(),
                      m_rectTee[1].output1());
    
    addProcess(&m_ctrlProc);
    addProcess(&m_rectify);
    addProcess(&m_rectTee[0]);
    addProcess(&m_rectTee[1]);
    addProcess(*m_matcher);    
}

void Reconstruction::dupRectficatin(ReadPipe<FloatImage> **leftRectOut, 
                                    ReadPipe<FloatImage> **rightRectOut)
{
    m_rectTee[0].enableOutput2();
    m_rectTee[1].enableOutput2();
    
    *leftRectOut = m_rectTee[0].output2();
    *rightRectOut = m_rectTee[1].output2();        
}

void Reconstruction::undupRectification()
{
    m_rectTee[0].disableOutput2();
    m_rectTee[1].disableOutput2();
}

TDV_NAMESPACE_END
