#include <gtest/gtest.h>
#include <tdvision/tdvision.hpp>
#include <cv.h>
#include <highgui.h>
#include "errorhandler.hpp"

class MedianFilterTest: public ::testing::Test
{
protected:
    MedianFilterTest()
        : reader("../../res/west.png"),
          writer("medianfilter_west.png")
    {
    }
    
    void SetUp()
    {                                
        fconv.input(reader.output());                                    
        writer.input(rconv.output());    
    }
    
    void TearDown()
    {
        
    }
    
    void TestFilter()
    {
    }
    
    tdv::ImageReader reader;
    tdv::FloatConv fconv;  
    tdv::MonoWorkUnit<tdv::FloatImage, tdv::FloatImage> *filter;
    tdv::RGBConv rconv;
    tdv::ImageWriter writer;
};


#if 0
TEST_F(MedianFilterTest, CPU)
{        
    filter = new tdv::MedianFilterCPU;
    
    filter->input(fconv.output());
    rconv.input(filter->output());
    
    reader.update();
    fconv.update();
    filter->update();
    rconv.update();
    writer.update();
}
#endif

TEST_F(MedianFilterTest, Dev)
{
    filter = new tdv::MedianFilterDev;
    
    filter->input(fconv.output());
    rconv.input(filter->output());

    tdv::WorkUnitProcess p0(reader);    
    tdv::CUDAProcess p1(0);
    tdv::WorkUnitProcess p2(writer);    
    
    p1.addWork(&fconv);
    p1.addWork(filter);
    p1.addWork(&rconv);
                   
    tdv::ArrayProcessGroup procs;    
    procs.addProcess(&p0);
    procs.addProcess(&p1);
    procs.addProcess(&p2);
    
    ErrorHandler errHdl;
    tdv::ProcessRunner runner(procs, &errHdl);    
    runner.run();
    runner.join();
    
    EXPECT_FALSE(runner.errorOcurred());    
}

