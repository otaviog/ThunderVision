#ifndef TDV_MUTUALINFORMATIONCPU_HPP
#define TDV_MUTUALINFORMATIONCPU_HPP

#include <tdvbasic/common.hpp>
#include "workunit.hpp"
#include "pipe.hpp"
#include "dsimem.hpp"
#include "floatimage.hpp"
#include "matchingcost.hpp"

TDV_NAMESPACE_BEGIN

class MutualInformationCPU: public MatchingCost
{
public:
    MutualInformationCPU(int disparityMax);
    
    void inputs(ReadPipe<FloatImage> *lpipe, ReadPipe<FloatImage> *rpipe)
    {
        m_lrpipe = lpipe;
        m_rrpipe = rpipe;
    }
    
    ReadPipe<DSIMem>* output()
    {
        return &m_wpipe;
    }
        
    bool update();
            
private:
    float mutualInformation(const float A[], const float B[], size_t sz, 
                            int histA[], int histB[], int histAB[], size_t histSize);

    void calcDSI(FloatImage limg_d, FloatImage rimg_d, float **dsiRet, Dim *dimRet);

    ReadPipe<FloatImage> *m_lrpipe, *m_rrpipe;
    ReadWritePipe<DSIMem, DSIMem> m_wpipe;
    size_t m_maxDisparaty;
};

TDV_NAMESPACE_END

#endif /* TDV_MUTUALINFORMATIONCPU_HPP */
