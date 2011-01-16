#include <gtest/gtest.h>

TEST(Image, Rationale)
{
    th::LumiImage image;
    float *gpuMem = image.lockGPU();
    cv::Mat mat = image.lockCPU();
    EXPECT_EQ(NULL, mat.refCount);

    image.unlockGPU();
    mat = image.lockCPU();
    EXPECT_NEQ(NULL, mat.refCount);


}
