#include <tdvision/stereomatcher.hpp>
#include <tdvision/imagewriter.hpp>
#include <tdvision/processrunner.hpp>
#include <highgui.h>
#include "ibenchmarkdataset.hpp"
#include "imatchercompmetric.hpp"
#include "benchmarkrunner.hpp"

TDV_NAMESPACE_BEGIN

BenchmarkRunner::BenchmarkRunner(
    StereoMatcher *matcher, IBenchmarkDataset *dataset,
    IMatcherCompMetric *metric)
{
    m_matcher = matcher;
    m_dataset = dataset;
    m_metric = metric;
}

void BenchmarkRunner::run()
{
    ReadWritePipe<FloatImage> leftPipe, rightPipe;

    m_matcher->inputs(&leftPipe, &rightPipe);

    ReadPipe<FloatImage> * const outPipe = m_matcher->output();
    for (size_t i=0; i<m_dataset->stereoPairCount(); i++)
    {
        StereoPair *spair = m_dataset->stereoPair(i);

        for (StereoPair::SampleList::iterator it = spair->samplesBegin();
             it != spair->samplesEnd(); it++)
        {
            StereoPair::Sample &sample(*it);

            leftPipe.write(sample.leftImage());
            rightPipe.write(sample.rightImage());

            leftPipe.finish();
            rightPipe.finish();

            ProcessRunner runner(*m_matcher, this);
            runner.run();
            runner.join();
            
            FloatImage matcherImage;
            if ( outPipe->read(&matcherImage) )
            {
                IMatcherCompMetric::Report report = m_metric->compare(
                    sample.groundTruth(), matcherImage);                
                cvSaveImage(
                    (boost::format("%1%_%2%x%3%_%4%.png")
                     % spair->name() % sample.width()
                     % sample.height() % m_matcher->name()).str().c_str(),
                    matcherImage.cpuMem());
            }
        }
    }
}

void BenchmarkRunner::errorOcurred(const std::exception &err)
{
}

TDV_NAMESPACE_END
