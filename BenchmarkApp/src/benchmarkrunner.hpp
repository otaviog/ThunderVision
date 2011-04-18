#ifndef TDV_BENCHMARKRUNNER_HPP
#define TDV_BENCHMARKRUNNER_HPP

#include <tdvbasic/common.hpp>

TDV_NAMESPACE_BEGIN

class StereoMatcher;
class IBenchmarkDataset;
class IMatcherCompMetric;

class BenchmarkRunner
{
public:
    BenchmarkRunner(StereoMatcher *matcher, IBenchmarkDataset *dataset,
                    IMatcherCompMetric *metric);
    
    void run();
    
private:
    StereoMatcher *m_matcher;
    IBenchmarkDataset *m_dataset;
    IMatcherCompMetric *m_metric;
};

TDV_NAMESPACE_END

#endif /* TDV_BENCHMARKRUNNER_HPP */
