#include <gtest/gtest.h>
#include <tdvision/mutualinformationdev.hpp>
#include <tdvision/mutualinformationcpu.hpp>
#include <tdvision/wtadev.hpp>
#include <tdvision/dynamicprogdev.hpp>
#include <tdvision/dynamicprogcpu.hpp>
#include "stereotest.hpp"

#if 0
TEST(TestMI, WithWTA)
{
    tdv::MutualInformationDev mi(128);
    tdv::WTADev wta;
    runStereoTest(
        "../../res/tsukuba512_L.png",
        "../../res/tsukuba512_R.png",
        "tsukuba_miwta.png", &mi, &wta);       
}

TEST(TestMI, WithDynProg)
{
    tdv::MutualInformationDev mi(128);
    tdv::DynamicProgDev dp;
    runStereoTest(
        "../../res/tsukuba512_L.png",
        "../../res/tsukuba512_R.png",
        "tsukuba_midynprog.png", &mi, &dp);   
}

TEST(TestMI, WithDynCPU)
{
    tdv::MutualInformationDev mi(128);
    tdv::DynamicProgCPU dp;
    runStereoTest(
        "../../res/tsukuba512_L.png",
        "../../res/tsukuba512_R.png",
        "tsukuba_midynprogcpu.png", &mi, &dp);
}

TEST(TestMI, WithMedianWTA)
{
    tdv::MutualInformationDev mi(128);
    tdv::WTADev wta;

    runStereoTest(
        "../../res/tsukuba512_L.png",
        "../../res/tsukuba512_R.png",
        "tsukuba_medianmiwta.png", &mi, &wta, true);   
}
#endif

TEST(TestMI, WithCPUWTA)
{
    tdv::MutualInformationCPU mi(128);
    tdv::WTADev wta;

    runStereoTest(
        "../../res/tsukuba512_L.png",
        "../../res/tsukuba512_R.png",
        "tsukuba_micpuwta.png", &mi, &wta, true);   
}
