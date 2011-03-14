#include <gtest/gtest.h>
#include <tdvision/benchmark.hpp>

TEST(TestBenchmark, ShouldMark)
{
    tdv::Timemark bmark0;
    tdv::Timemark bmark1;
    tdv::Benchmarker marker;
    
    bmarker0.addProbeSec(1.0);
    
    marker.begin(bmark0);
    usleep(1000);
    marker.end();
    
    EXPECT_DOUBLE_EQ(1.0, bmark0.secs());
    
    tdv::Benchmark bmark1;    
    for (size_t i=0; i<1000; i++)
    {
        marker.begin(bmark1);
        usleep(1);
        marker.end();
    }
    
    EXPECT_DOUBLE_EQ(0.001, bmark1.secs());    
}

TEST(TestBenchmark, ShouldBeOnSuite)
{
    tdv::Timemark bmark0;
    tdv::Timemark bmark1;
    
    tdv::Benchmarker marker;
        
    marker.begin(bmark0);
    usleep(2);
    marker.end();
    
    marker.begin(bmark1);
    usleep(3);
    marker.end();

    EXPECT_DOUBLE_EQ(0.002, bmark0.secs());
    EXPECT_DOUBLE_EQ(0.003, bmark1.secs());    
    
    tdv::BenchmarkSuite bsuite;
    bsuite.set("Filter 0", bmark0);
    bsuite.set("Filter 1", bmark1);
    
    EXPECT_EQ(2, bsuite.count());    
}
