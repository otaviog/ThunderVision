#include <gtest/gtest.h>
#include <tdvision/imagereader.hpp>
#include <tdvision/imagewriter.hpp>
#include <tdvision/cpyimagetocpu.hpp>
#include <tdvision/ssddev.hpp>
#include <tdvision/wtadev.hpp>
#include <tdvision/medianfilterdev.hpp>

#include <cv.h>
#include <highgui.h>

TEST(TestSSD, Dev)
{
    tdv::ImageReader readerL("../../res/tsukuba_L.png");
    tdv::ImageReader readerR("../../res/tsukuba_R.png");
    
    tdv::SSDDev ssd(155, 1024*1024*128);
    
    ssd.leftImageInput(readerL.output());
    ssd.rightImageInput(readerR.output());    
    
    readerL.update();
    readerR.update();
    ssd.update();
    
    tdv::DSIMem dsi;
    ASSERT_TRUE(ssd.output()->read(&dsi));
    
    EXPECT_EQ(384, dsi.dim().width());
    EXPECT_EQ(288, dsi.dim().height());
    EXPECT_EQ(155, dsi.dim().depth());    
}

TEST(TestSSD, WithWTA)
{
    tdv::ImageReader readerL("../../res/tsukuba512_L.png");
    tdv::ImageReader readerR("../../res/tsukuba512_R.png");
    
    tdv::SSDDev ssd(155, 1024*1024*128);
    tdv::WTADev wta;
    
    tdv::ImageWriter writer("tsukuba_ssdwta.png");
    
    ssd.leftImageInput(readerL.output());
    ssd.rightImageInput(readerR.output());    
    wta.input(ssd.output());
    writer.input(wta.output());
    
    readerL.update();
    readerR.update();
    ssd.update();
    wta.update();
    writer.update();
    
    tdv::FloatImage dispimg;
    ASSERT_TRUE(writer.output()->read(&dispimg));    

    dispimg.dispose();
}

TEST(TestSSD, WithMedianWTA)
{
    tdv::ImageReader readerL("../../res/tsukuba512_L.png");
    tdv::ImageReader readerR("../../res/tsukuba512_R.png");
    tdv::MedianFilterDev mfL, mfR;
    
    tdv::SSDDev ssd(155, 1024*1024*128);
    tdv::WTADev wta;
    
    tdv::ImageWriter writer("tsukuba_medianssdwta.png");
    
    mfL.input(readerL.output());
    mfR.input(readerR.output());
    ssd.leftImageInput(mfL.output());
    ssd.rightImageInput(mfR.output());    
    wta.input(ssd.output());
    writer.input(wta.output());
    
    readerL.update();
    readerR.update();
    mfL.update();
    mfR.update();
    ssd.update();
    wta.update();
    writer.update();
    
    tdv::FloatImage dispimg;
    ASSERT_TRUE(writer.output()->read(&dispimg));    
    
    dispimg.dispose();
}

