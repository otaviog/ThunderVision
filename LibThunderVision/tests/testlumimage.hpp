#include <gtest/gtest.h>
#include "lumimage.hpp"

TEST(LumImage, Rationalle)
{
    th::LumImage image;
    EXPECT_TRUE(image.load("res/taksuba_left.png"));

    EXPECT_EQ(256, image.width());
    EXPECT_EQ(256, image.height());
    th::LumImage image2 = image;
    image.
    cv::Mat mat = image.lockCPU();
    image.unlockCPU();


}
