#include <gtest/gtest.h>
#include <tdvbasic/log.hpp>
#include <tdvision/imagereader.hpp>
#include <tdvision/imagewriter.hpp>
#include <tdvision/cpyimagetocpu.hpp>
#include <tdvision/crosscorrelationdev.hpp>
#include <tdvision/wtadev.hpp>
#include <tdvision/medianfilterdev.hpp>
#include <tdvision/medianfiltercpu.hpp>
#include <tdvision/floatconv.hpp>
#include <tdvision/rgbconv.hpp>
#include <tdvision/dynamicprogdev.hpp>
#include <tdvision/dynamicprogcpu.hpp>
#include <cv.h>
#include <highgui.h>

TEST(TestCrossCorrelation, Dev)
{
    tdv::TdvGlobalLogDefaultOutputs();
    
    tdv::ImageReader readerL("../../res/tsukuba_L.png");
    tdv::ImageReader readerR("../../res/tsukuba_R.png");
    tdv::FloatConv fconvl, fconvr;
    tdv::RGBConv rconvl, rconvr;
    
    tdv::CrossCorrelationDev xcorr(16);
    
    fconvl.input(readerL.output());
    fconvr.input(readerR.output());
    
    xcorr.inputs(fconvl.output(), fconvr.output());    
    
    readerL.update();
    readerR.update();
    fconvl.update();
    fconvr.update();
    xcorr.update();
    
    tdv::DSIMem dsi;
    ASSERT_TRUE(xcorr.output()->read(&dsi));
    
    EXPECT_EQ(384, dsi.dim().width());
    EXPECT_EQ(288, dsi.dim().height());
    EXPECT_EQ(16, dsi.dim().depth());    
}

static void runOptimizerTest(const std::string &outputImg, tdv::Optimizer *opt)
{
    tdv::ImageReader readerL("../../res/tsukuba512_L.png");
    tdv::ImageReader readerR("../../res/tsukuba512_R.png");
    tdv::FloatConv fconvl, fconvr;
    tdv::RGBConv rconv;
    
    tdv::CrossCorrelationDev xcorr(128);    
    tdv::ImageWriter writer(outputImg);
    tdv::MedianFilterCPU ml, mr;
    
    fconvl.input(readerL.output());
    fconvr.input(readerR.output());
    
    ml.input(fconvl.output());
    mr.input(fconvr.output());
    
    //ssd.inputs(fconvl.output(), fconvr.output());    
    xcorr.inputs(ml.output(), mr.output());    
    
    opt->input(xcorr.output());    
    rconv.input(opt->output());
    writer.input(rconv.output());
    
    readerL.update();
    readerR.update();
    fconvl.update();
    fconvr.update();
    ml.update();
    mr.update();
    xcorr.update();
    opt->update();
    rconv.update();
    writer.update();    
}

TEST(TestXCorr, WithWTA)
{
    tdv::WTADev wta;
    runOptimizerTest("tsukuba_xcorrwta.png", &wta);       
}

TEST(TestXCorr, WithDynProg)
{
    tdv::DynamicProgDev dp;
    runOptimizerTest("tsukuba_xcorrdynprog.png", &dp);   
}

TEST(TestXCorr, WithDynCPU)
{
    tdv::DynamicProgCPU dp;
    runOptimizerTest("tsukuba_xcorrdynprogcpu.png", &dp);
}
