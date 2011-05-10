#include <tdvision/stereomatcher.hpp>
#include <tdvision/imagewriter.hpp>
#include <tdvision/processrunner.hpp>
#include <highgui.h>
#include <iostream>
#include "bmdataset.hpp"
#include "qualitymetric.hpp"
#include "bmrunner.hpp"

TDV_NAMESPACE_BEGIN

BMRunner::BMRunner(
    StereoMatcher *matcher, BMDataset *dataset,
    QualityMetric *metric)
{
    m_matcher = matcher;
    m_dataset = dataset;
    m_metric = metric;
    m_hasError = false;
}

bool BMRunner::run()
{
    using namespace bmdata;
    
    ReadWritePipe<FloatImage> leftPipe, rightPipe;

    m_matcher->inputs(&leftPipe, &rightPipe);

    ReadPipe<FloatImage> * const outPipe = m_matcher->output();
    for (size_t i=0; i<m_dataset->stereoPairCount(); i++)
    {
        StereoPair *spair = m_dataset->stereoPair(i);
        bmdata::StereoPairReport report(spair->name(), m_matcher->name());
        
        for (StereoPair::SampleList::iterator it = spair->samplesBegin();
             it != spair->samplesEnd(); it++)
        {
            Sample &sample(*it);

            leftPipe.write(sample.leftImage());
            rightPipe.write(sample.rightImage());

            leftPipe.finish();
            rightPipe.finish();

            ProcessRunner runner(*m_matcher, this);
            runner.run();
            runner.join();
            
            if ( m_hasError )
                return false;
            
            FloatImage matcherImage;
            if ( outPipe->read(&matcherImage) )
            {
                const QualityMetric::Report qrep = m_metric->compare(
                    sample.groundTruth(), matcherImage);
                cvConvertScale(matcherImage.cpuMem(), matcherImage.cpuMem(), 
                               255.0);
                cvSaveImage(
                    (boost::format("%1%_%2%x%3%_%4%.png")
                     % spair->name() % sample.dim().width()
                     % sample.dim().height() % m_matcher->name()).str().c_str(),
                    matcherImage.cpuMem());                                
                
                // report.addSampleReport(
                //     SampleReport(qrep.error, 
                //                  m_matcher->benchmark().secs(), 
                //                  sample.dim()));                                
            }            
        }
        
        m_reports.push_back(report);
    }
    
    return true;
}

void BMRunner::errorOcurred(const std::exception &err)
{
    std::cout << err.what() << std::endl;
    m_hasError = true;
}

TDV_NAMESPACE_END
