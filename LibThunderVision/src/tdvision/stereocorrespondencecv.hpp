#ifndef TDV_STEREOCORRESPONDENCECV_HPP
#define TDV_STEREOCORRESPONDENCECV_HPP

#include <tdvbasic/common.hpp>
#include <cv.h>
#include "pipe.hpp"
#include "workunit.hpp"
#include "tmpbufferimage.hpp"
#include "floatimage.hpp"

TDV_NAMESPACE_BEGIN

class StereoCorrespondenceCV: public WorkUnit
{
public:
    enum MatchingMode
    {
        LocalMatching,
        GlobalMatching
    };
    
    StereoCorrespondenceCV(MatchingMode mode, 
                           int maxDisparity, int maxIterations = 4);

    ~StereoCorrespondenceCV();

    bool update();
    
    void inputs(ReadPipe<FloatImage> *leftInput,
                ReadPipe<FloatImage> *rightInput)
    {
        m_lrpipe = leftInput;
        m_rrpipe = rightInput;
    }

    ReadPipe<FloatImage>* output()
    {
        return &m_wpipe;
    }

private:
    ReadPipe<FloatImage> *m_lrpipe, *m_rrpipe;
    ReadWritePipe<FloatImage> m_wpipe;
    MatchingMode m_mode;
    
    TmpBufferImage m_limg8U, m_rimg8U;
    union 
    {
        CvStereoBMState *m_bmState;
        CvStereoGCState *m_gcState;
    };
};

TDV_NAMESPACE_END

#endif /* TDV_STEREOCORRESPONDENCECV_HPP */
