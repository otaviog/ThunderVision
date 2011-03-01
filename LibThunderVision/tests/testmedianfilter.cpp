#include <gtest/gtest.h>
#include <tdvision/medianfilterwucpu.hpp>
#include <tdvision/medianfilterwudev.hpp>
#include <cv.h>
#include <highgui.h>

TEST(MedianFilterTest, CPU)
{    
    tdv::FloatImage image(cvLoadImage("../../res/west.png"));
    
    tdv::MedianFilterWUCPU mfilter;    
    tdv::ReadWritePipe<tdv::FloatImage, tdv::FloatImage> inpipe, outpipe;
    
    EXPECT_NO_THROW({            
            mfilter.input(&inpipe);
            mfilter.output(&outpipe);
            inpipe.write(image);    
            inpipe.end();
            mfilter.process();

            image.dispose();
        });
    
    outpipe.waitPacket();    
    tdv::FloatImage output = outpipe.read();

    cvSaveImage("west_out.jpg", output.waitCPUMem());
    cvShowImage("image", output.waitCPUMem());
    cvWaitKey(0);

    output.dispose();    
}

TEST(MedianFilterTest, Dev)
{
    tdv::FloatImage image(cvLoadImage("../../res/west.png"));
    tdv::MedianFilterWUDev mfilter;        
    tdv::ReadWritePipe<tdv::FloatImage, tdv::FloatImage> inpipe, outpipe;
    tdv::FloatImage output;
                    
    mfilter.input(&inpipe);
    mfilter.output(&outpipe);
        
    inpipe.write(image);
    inpipe.end();
        
    mfilter.process();              
    outpipe.waitPacket();
    output = outpipe.read();
        
    image.dispose();
                
    cvSaveImage("west_out.png", output.waitCPUMem());
    cvShowImage("image", output.waitCPUMem());
    
    cvWaitKey(0);
    output.dispose();
}

TEST(MedianFilterTest, CPUDevDevCPU)
{

}
