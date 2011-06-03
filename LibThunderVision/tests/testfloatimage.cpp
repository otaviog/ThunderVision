#include <gtest/gtest.h>
#include <tdvision/floatimage.hpp>
#include <highgui.h>
#include "errorhandler.hpp"

TEST(TestFloatImage, Copy)
{
    IplImage *img = cvLoadImage("img.png");
    ASSERT_TRUE(img != NULL);
    
    cvShowImage("Origin", img);
    
    tdv::FloatImage flimg(img);
    
    float *device = flimg.devMem();    
    CvMat *cpu = flimg.cpuMem();
    
    cvShowImage("CPU", cpu);
    cvWaitKey(0);
}
