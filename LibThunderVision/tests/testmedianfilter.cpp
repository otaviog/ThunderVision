#include <gtest/gtest.h>
#include <tdvision/medianfilterwucpu.hpp>
#include <tdvision/medianfilterwudev.hpp>
#include <tdvision/imagereaderwu.hpp>
#include <tdvision/imagewriterwu.hpp>
#include <tdvision/workunitrunner.hpp>
#include <cv.h>
#include <highgui.h>

class ErrorHandle: public tdv::WorkExceptionReport
{
public:
    void errorOcurred(const std::exception &err)
    {
        std::cout<<err.what()<<std::endl;
    }
};

TEST(MedianFilterTest, CPU)
{    
    tdv::ImageReaderWU reader("../../res/west.png");
    tdv::MedianFilterWUCPU mfilter;        
    tdv::ImageWriterWU writer("../../res/medianfilter_west.png");
    
    mfilter.input(reader.output());    
    writer.input(mfilter.output());
    
    tdv::WorkUnit *wus[] = {&reader, &mfilter, &writer};
    ErrorHandle errHdl;
    tdv::WorkUnitRunner runner(wus, 3, &errHdl);
    
    runner.run();
    runner.join();
    
    EXPECT_FALSE(runner.errorOcurred());
    
    if ( !runner.errorOcurred() )
    {
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
}

TEST(MedianFilterTest, Dev)
{
    tdv::ImageReaderWU reader("../../res/west.png");
    tdv::MedianFilterWUDev mfilter;        
    tdv::ImageWriterWU writer("../../res/medianfilter_west.png");
    
    mfilter.input(reader.output());    
    writer.input(mfilter.output());
    
    tdv::WorkUnit *wus[] = {&reader, &mfilter, &writer};
    ErrorHandle errHdl;
    tdv::WorkUnitRunner runner(wus, 3, &errHdl);
    
    runner.run();
    runner.join();
    
    EXPECT_FALSE(runner.errorOcurred());
    
    if ( !runner.errorOcurred() )
    {
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
}

TEST(MedianFilterTest, CPUDevDevCPU)
{

}
