#include <gtest/gtest.h>
#include <tdvision/imagereader.hpp>
#include <tdvision/imagewriter.hpp>
#include <tdvision/cpyimagetocpu.hpp>
#include <tdvision/ssddev.hpp>
#include <tdvision/wtadev.hpp>
#include <tdvision/medianfilterdev.hpp>
#include <tdvision/floatconv.hpp>
#include <tdvision/rgbconv.hpp>
#include <cv.h>
#include <highgui.h>

TEST(TestSSD, Dev)
{
    tdv::ImageReader readerL("../../res/tsukuba_L.png");
    tdv::ImageReader readerR("../../res/tsukuba_R.png");
    tdv::FloatConv fconvl, fconvr;
    tdv::RGBConv rconvl, rconvr;
    
    tdv::SSDDev ssd(155, 1024*1024*128);
    
    fconvl.input(readerL.output());
    fconvr.input(readerR.output());
    
    ssd.inputs(fconvl.output(), fconvr.output());    
    
    readerL.update();
    readerR.update();
    fconvl.update();
    fconvr.update();
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
    tdv::FloatConv fconvl, fconvr;
    tdv::RGBConv rconv;
    
    tdv::SSDDev ssd(155, 1024*1024*128);
    tdv::WTADev wta;
    
    tdv::ImageWriter writer("tsukuba_ssdwta.png");
    
    fconvl.input(readerL.output());
    fconvr.input(readerR.output());
    ssd.inputs(fconvl.output(), fconvr.output());    
    wta.input(ssd.output());    
    rconv.input(wta.output());
    writer.input(rconv.output());
    
    readerL.update();
    readerR.update();
    fconvl.update();
    fconvr.update();
    ssd.update();
    wta.update();
    rconv.update();
    writer.update();    
}

TEST(TestSSD, WithMedianWTA)
{
    tdv::ImageReader readerL("../../res/tsukuba512_L.png");
    tdv::ImageReader readerR("../../res/tsukuba512_R.png");
    tdv::FloatConv fconvl, fconvr;
    tdv::MedianFilterDev mfL, mfR;    
    tdv::SSDDev ssd(155, 1024*1024*128);
    tdv::WTADev wta;
    tdv::RGBConv rconv;    
    
    tdv::ImageWriter writer("tsukuba_medianssdwta.png");

    fconvl.input(readerL.output());
    fconvr.input(readerR.output());    
    mfL.input(fconvl.output());
    mfR.input(fconvr.output());
    ssd.inputs(mfL.output(), mfR.output());    
    wta.input(ssd.output());
    rconv.input(wta.output());
    writer.input(rconv.output());
    
    readerL.update();
    readerR.update();
    fconvl.update();
    fconvr.update();
    mfL.update();
    mfR.update();
    ssd.update();
    wta.update();
    rconv.update();
    writer.update();    
}

