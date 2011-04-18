#include <tdvision/stereomatcher.hpp>
#include <tdvision/imagewriter.hpp>
#include <tdvision/processrunner.hpp>
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
    ImageWriter writer("");
    
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
            
            ProcessRunner runner(*m_stereoMatcher);
            runner.run();            
            runner.join();            
            
            FloatImage matcherImage;
            if ( outPipe->read(&matcherImage) )
            {                
                IMatcherCompMetric::Report report = m_metric->compare(
                    sample.groundTruth(), matcherImage);
            
                writer.setFilename((boost::format("%1%_%2%x%3%_%4%") 
                                    % spair->name() % sample.width() 
                                    % sample.height() % m_stereoMatcher->name()).str());                        
            }
        }
    }

    
}

TDV_NAMESPACE_END
