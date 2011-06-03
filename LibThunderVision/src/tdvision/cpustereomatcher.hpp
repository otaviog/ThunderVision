#ifndef TDV_CPUSTEREOMATCHER_HPP
#define TDV_CPUSTEREOMATCHER_HPP

#include <tdvbasic/common.hpp>
#include "workunitprocess.hpp"
#include "processgroup.hpp"
#include "stereomatcher.hpp"
#include "stereocorrespondencecv.hpp"

TDV_NAMESPACE_BEGIN

class CPUStereoMatcher: public StereoMatcher
{
public:
    CPUStereoMatcher(StereoCorrespondenceCV::MatchingMode mode,
                     int maxDisparity, int maxIteration);

    void inputs(ReadPipe<FloatImage> *leftInput,
                ReadPipe<FloatImage> *rightInput)
    {
        m_corresp.inputs(leftInput, rightInput);
    }

    ReadPipe<FloatImage>* output()
    {
        return m_corresp.output();
    }

    Process** processes()
    {
        return m_procs.processes();
    }
    
    std::string name() const
    {        
        return "CPU+" + m_corresp.workName();
    }
    
    Benchmark matchcostBenchmark() const
    {
        return m_corresp.benchmark();
    }
    
    Benchmark optimizationBenchmark() const
    {
        return m_corresp.benchmark();
    }

private:
    StereoCorrespondenceCV m_corresp;
    PWorkUnitProcess m_correspProc;
    ArrayProcessGroup m_procs;
};

TDV_NAMESPACE_END

#endif /* TDV_CPUSTEREOMATCHER_HPP */
