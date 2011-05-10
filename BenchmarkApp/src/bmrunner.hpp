#ifndef TDV_BMRUNNER_HPP
#define TDV_BMRUNNER_HPP

#include <tdvbasic/common.hpp>
#include <tdvision/exceptionreport.hpp>

TDV_NAMESPACE_BEGIN

class StereoMatcher;
class BMDataset;
class QualityMetric;

class BMRunner: public ExceptionReport
{
public:
    BMRunner(StereoMatcher *matcher, BMDataset *dataset,
                    QualityMetric *metric);
    
    bool run();
    
    void errorOcurred(const std::exception &err);
    
private:
    StereoMatcher *m_matcher;
    BMDataset *m_dataset;
    QualityMetric *m_metric;
    bool m_hasError;
    
    std::vector<bmdata::StereoPairReport> m_reports;
};

TDV_NAMESPACE_END

#endif /* TDV_BMRUNNER_HPP */
