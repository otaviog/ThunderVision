#include <gtest/gtest.h>
#include <tdvision/medianfiltercpu.hpp>
#include <tdvision/medianfilterdev.hpp>
#include <tdvision/imagereader.hpp>
#include <tdvision/imagewriter.hpp>
#include <tdvision/cpyimagetocpu.hpp>
#include <tdvision/processgroup.hpp>
#include <tdvision/processrunner.hpp>
#include <tdvision/workunitprocess.hpp>
#include <tdvision/cudaprocess.hpp>
#include <cv.h>
#include <highgui.h>
#include "errorhandler.hpp"

TEST(MedianFilterTest, CPU)
{    
    tdv::ImageReader reader("../../res/west.png");
    tdv::MedianFilterDev mfilter;        
    tdv::ImageWriter writer("../../res/medianfilter_west.png");
    
    mfilter.input(reader.output());    
    writer.input(mfilter.output());    
    
    reader.update();
    mfilter.update();
    writer.update();
        
    tdv::FloatImage output;
    bool read;
    read = writer.output()->read(&output);
        
    EXPECT_TRUE(read);
        
    if ( read )
    {
        cvShowImage("image", output.cpuMem());
        cvWaitKey(0);
        
        output.dispose();
    }   
}

TEST(MedianFilterTest, Dev)
{
    tdv::ImageReader reader("../../res/west.png");
    tdv::MedianFilterDev mfilter;        
    tdv::CpyImageToCPU mconv;
    tdv::ImageWriter writer("../../res/medianfilter_west.png");

    mfilter.input(reader.output());    
    mconv.input(mfilter.output());
    writer.input(mconv.output());
    
    tdv::WorkUnitProcess p0(reader);
    tdv::WorkUnitProcess p1(writer);
    tdv::CUDAProcess p2(0);
    
    p2.addWork(&mfilter);
    p2.addWork(&mconv);
               
    tdv::Process *s[] = {&p0, &p1, &p2, NULL};
    ErrorHandler errHdl;
    tdv::ArrayProcessGroup procs(s);
    tdv::ProcessRunner runner(procs, &errHdl);    
    runner.run();
    runner.join();
    
    EXPECT_FALSE(runner.errorOcurred());
    
    if ( !runner.errorOcurred() )
    {
        tdv::FloatImage output;
        bool read;
        read = mfilter.output()->read(&output);
        
        EXPECT_TRUE(read);
        
        if ( read )
        {
            cvShowImage("image", output.cpuMem());
            cvWaitKey(0);
            
            output.dispose();
        }
    }        
}

TEST(MedianFilterTest, CPUDevDevCPU)
{
    
}
