#include "reconstructcontext.hpp"

ReconstructContext::init()
{        
    m_rectify.leftImgInput(m_leftImg);    
    m_rectify.rightImgInput(m_rightImg);    
    
    m_floatConv[0].input(m_rectify.leftImgOutput());
    m_floatConv[1].input(m_rectify.rightImgOutput());
        
    assert(m_preFilters.size() % 2 == 0);        
    
    tdv::ReadPipe<FloatImage> *lastLeftPipe = m_floatConv[0].output();
    tdv::ReadPipe<FloatImage> *lastRightPipe = m_floatConv[1].output();
    
    for (size_t i=0; i<m_preFilters.size(); i += 2)
    {
        m_preFilters[i]->input(lastLeftPipe);
        m_preFilters[i + 1]->input(lastRightPipe);
        
        lastLeftPipe = m_preFilters[i].output();
        lastRightPipe = m_preFilters[i + 1].output();        
    }
    
    m_stereoMatcher.leftImgInput(lastLeftPipe);
    m_stereoMatcher.rightImgInput(lastRightPipe);        
    
    //m_stereoMatcher.
}
