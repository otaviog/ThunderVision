#ifndef TDV_BENCHMARKRUNNER_HPP
#define TDV_BENCHMARKRUNNER_HPP

#include <tdvbasic/common.hpp>
#include <tdvision/exceptionreport.hpp>

TDV_NAMESPACE_BEGIN

class StereoMatcher;
class IBenchmarkDataset;
class IMatcherCompMetric;

class BenchmarkRunner: public ExceptionReport
{
public:
    BenchmarkRunner(StereoMatcher *matcher, IBenchmarkDataset *dataset,
                    IMatcherCompMetric *metric);
    
    void run();
    
    void errorOcurred(const std::exception &err);
    
private:
    StereoMatcher *m_matcher;
    IBenchmarkDataset *m_dataset;
    IMatcherCompMetric *m_metric;
};

TDV_NAMESPACE_END

#endif /* TDV_BENCHMARKRUNNER_HPP */
