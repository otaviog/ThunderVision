#include <gtest/gtest.h>

/**
 * Given a StereoMatcher
 * I should benchmark it.
 * I must have a dataset to benchmark it against.
 * Each member of dataset have a pair of left and right images
 * and a true disparity image
 */
TEST(SpecBenchmark, GivenAStereoMatcher)
{
    IBenchmarkDataset *dataset = DatasetFactory::CreateDefaultSet();    
    StereoPair *stereoPairSet = dataset->imageSet("teddy");
    
    for (StereoPair::PairList::iterator it = stereoPairSet->resolutions().begin(); 
         it != stereoPairSet->resolutions().end(); it++)
    {
        StereoPair::Pair pair(it);
        pair.leftImage();
        pair.rightImage();
        pair.groundThruth();                
        pair.width();
        pair.height();
    }
}

/**
 * Then I need run the StereoMatcher against the Dataset,
 * With a metric to mesure how good is my stereo matcher 
 * And then I need output the results.
 */
TEST(SpecBenchmark, ShouldRunStereoMatcher)
{
    CPUStereoMatcherFactory factory;
    StereoMatcher *matcher = factory.create();    
    IBenchmarkDataset *dataset = DatasetFactory::CreateDefaultSet();
    
    IMatcherCompMetric *metric = new PixelMedianCompMetric;
    
    BenchmarkRunner runner(dataset, matcher, metric);    
    BenchmarkDatasetResult results = runner.benchmark();    
    
    for (size_t i=0; i<results.stereoPairCount(); i++)
    {
        BenchmarkStereoPairResult result(results.stereoPair(i));
        
        for (size_t r=0; r<result.resolutions(); r++)        
        {
            tdv::Benchmark bench = result.benchmarkAt(r);
            result.resAt(r);            
        }
    }        
}

