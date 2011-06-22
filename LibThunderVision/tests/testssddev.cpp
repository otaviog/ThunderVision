#include <gtest/gtest.h>
#include <tdvision/ssddev.hpp>
#include <tdvision/wtadev.hpp>
#include <tdvision/imagereader.hpp>
#include <tdvision/floatconv.hpp>
#include <tdvision/rgbconv.hpp>
#include <tdvision/dynamicprogdev.hpp>
#include <tdvision/dynamicprogcpu.hpp>
#include "stereotest.hpp"

#if 0
TEST(TestSSD, Dev)
{    
    tdv::ImageReader readerL("../../res/tsukuba_L.png");
    tdv::ImageReader readerR("../../res/tsukuba_R.png");
    tdv::FloatConv fconvl, fconvr;
    tdv::RGBConv rconvl, rconvr;
    
    tdv::SSDDev ssd(16);
    
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
    EXPECT_EQ(16, dsi.dim().depth());    
}
#endif

TEST(TestSSD, WithWTA)
{
   tdv::SSDDev ssd(128);
   tdv::WTADev wta;
    
   runStereoTest(
       "../../res/tsukuba512_L.png",
       "../../res/tsukuba512_R.png",       
        "tsukuba_ssdwta.png",
       &ssd, &wta);
}

#if 0
TEST(TestSSD, WithDynProg)
{
    tdv::DynamicProgDev dp;    
    tdv::SSDDev ssd(128);
    runStereoTest(
        "../../res/tsukuba512_L.png",
        "../../res/tsukuba512_R.png",
        "tsukuba_ssddynprog.png",
        &ssd, &dp);
}

TEST(TestSSD, WithDynCPU)
{
    tdv::SSDDev ssd(128);
    tdv::DynamicProgCPU dp;
    runStereoTest(
        "../../res/tsukuba512_L.png",
        "../../res/tsukuba512_R.png",
        "tsukuba_ssddynprogcpu.png", 
        &ssd, &dp);
}

TEST(TestSSD, WithMedianWTA)
{
    tdv::SSDDev ssd(155);
    tdv::WTADev wta;
    
    runStereoTest(
        "../../res/tsukuba512_L.png",
        "../../res/tsukuba512_R.png",
        "tsukuba_medianssdwta.png",
        &ssd, &wta, true);
}
#endif
