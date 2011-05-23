#include <gtest/gtest.h>
#include <tdvision/crosscorrelationdev.hpp>
#include <tdvision/wtadev.hpp>
#include <tdvision/dynamicprogdev.hpp>
#include <tdvision/dynamicprogcpu.hpp>
#include "stereotest.hpp"

TEST(TestXCorr, WithWTA)
{
    tdv::CrossCorrelationDev xcorr(128);
    tdv::WTADev wta;    
    runStereoTest(
        "../../res/tsukuba512_L.png",
        "../../res/tsukuba512_R.png",
        "tsukuba_xcorrwta.png", &xcorr, &wta);       
}

TEST(TestXCorr, WithDynProg)
{
    tdv::CrossCorrelationDev xcorr(128);    
    tdv::DynamicProgDev dp;
    runStereoTest(
        "../../res/tsukuba512_L.png",
        "../../res/tsukuba512_R.png",
        "tsukuba_xcorrdynprog.png", &xcorr, &dp);   
}

TEST(TestXCorr, WithDynCPU)
{
    tdv::CrossCorrelationDev xcorr(128);
    tdv::DynamicProgCPU dp;
    runStereoTest(
        "../../res/tsukuba512_L.png",
        "../../res/tsukuba512_R.png",
        "tsukuba_xcorrdynprogcpu.png", &xcorr, &dp);
}
