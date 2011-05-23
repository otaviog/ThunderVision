#include <gtest/gtest.h>
#include <tdvbasic/log.hpp>
#include <tdvision/imagereader.hpp>
#include <tdvision/imagewriter.hpp>
#include <tdvision/cpyimagetocpu.hpp>
#include <tdvision/mutualinformationdev.hpp>
#include <tdvision/wtadev.hpp>
#include <tdvision/medianfilterdev.hpp>
#include <tdvision/medianfiltercpu.hpp>
#include <tdvision/floatconv.hpp>
#include <tdvision/rgbconv.hpp>
#include <tdvision/dynamicprogdev.hpp>
//#include <tdvision/dynamicprogcpu.hpp>
#include <cv.h>
#include <highgui.h>

TEST(TestMI, Dev)
{
    tdv::TdvGlobalLogDefaultOutputs();
    
    tdv::ImageReader readerL("../../res/tsukuba_L.png");
    tdv::ImageReader readerR("../../res/tsukuba_R.png");
    tdv::FloatConv fconvl, fconvr;
    tdv::RGBConv rconvl, rconvr;
    
    tdv::MutualInformationDev mi(16);
    
    fconvl.input(readerL.output());
    fconvr.input(readerR.output());
    
    mi.inputs(fconvl.output(), fconvr.output());    
    
    readerL.update();
    readerR.update();
    fconvl.update();
    fconvr.update();
    mi.update();
    
    tdv::DSIMem dsi;
    ASSERT_TRUE(mi.output()->read(&dsi));
    
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
    
    tdv::MutualInformationDev mi(200);    
    tdv::ImageWriter writer(outputImg);
    tdv::MedianFilterCPU ml, mr;
    
    fconvl.input(readerL.output());
    fconvr.input(readerR.output());
    
    ml.input(fconvl.output());
    mr.input(fconvr.output());
    
    //mi.inputs(fconvl.output(), fconvr.output());    
    mi.inputs(ml.output(), mr.output());    
    
    opt->input(mi.output());    
    rconv.input(opt->output());
    writer.input(rconv.output());
    
    readerL.update();
    readerR.update();
    fconvl.update();
    fconvr.update();
    ml.update();
    mr.update();
    mi.update();
    opt->update();
    rconv.update();
    writer.update();    
}


TEST(TestMI, WithWTA)
{
    tdv::WTADev wta;
    runOptimizerTest("tsukuba_miwta.png", &wta);       
}

#if 0
TEST(TestMI, WithDynProg)
{
    tdv::DynamicProgDev dp;
    runOptimizerTest("tsukuba_midynprog.png", &dp);   
}

TEST(TestMI, WithDynCPU)
{
    tdv::DynamicProgCPU dp;
    runOptimizerTest("tsukuba_midynprogcpu.png", &dp);
}
#endif

TEST(TestMI, WithMedianWTA)
{
    tdv::ImageReader readerL("../../res/tsukuba512_L.png");
    tdv::ImageReader readerR("../../res/tsukuba512_R.png");
    tdv::FloatConv fconvl, fconvr;
    //tdv::MedianFilterDev mfL, mfR;    
    tdv::MutualInformationDev mi(155);
    tdv::WTADev wta;
    tdv::RGBConv rconv;    
    
    tdv::ImageWriter writer("tsukuba_medianmiwta.png");

    fconvl.input(readerL.output());
    fconvr.input(readerR.output());    
    //mfL.input(fconvl.output());
    //mfR.input(fconvr.output());
    //mi.inputs(mfL.output(), mfR.output());    
    mi.inputs(fconvl.output(), fconvr.output());    
    wta.input(mi.output());
    rconv.input(wta.output());
    writer.input(rconv.output());
    
    readerL.update();
    readerR.update();
    fconvl.update();
    fconvr.update();
    //mfL.update();
    //mfR.update();
    mi.update();
    wta.update();
    rconv.update();
    writer.update();    
}
