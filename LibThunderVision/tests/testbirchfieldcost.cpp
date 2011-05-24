#include <gtest/gtest.h>
#include <tdvision/birchfieldcostdev.hpp>
#include <tdvision/wtadev.hpp>
#include <tdvision/dynamicprogdev.hpp>
#include <tdvision/dynamicprogcpu.hpp>
#include "stereotest.hpp"

TEST(TestBirchfieldCost, WithWTA)
{
    tdv::BirchfieldCostDev bf(64);
    tdv::WTADev wta;
    runStereoTest(
        "../../res/tsukuba512_L.png",
        "../../res/tsukuba512_R.png",
        "tsukuba_birchfieldwta.png", &bf, &wta);       
}

TEST(TestBirchfieldCost, WithDynProg)
{
    tdv::BirchfieldCostDev bf(64);
    tdv::DynamicProgDev dp;
    runStereoTest(
        "../../res/tsukuba512_L.png",
        "../../res/tsukuba512_R.png",
        "tsukuba_birchfielddynprog.png", &bf, &dp);   
}

TEST(TestBirchfieldCost, WithDynCPU)
{
    tdv::BirchfieldCostDev bf(64);
    tdv::DynamicProgCPU dp;
    runStereoTest(
        "../../res/tsukuba512_L.png",
        "../../res/tsukuba512_R.png",
        "tsukuba_birchfielddynprogcpu.png", &bf, &dp);
}

TEST(TestBirchfieldCost, WithMedianWTA)
{
    tdv::BirchfieldCostDev bf(64);
    tdv::WTADev wta;

    runStereoTest(
        //"../../res/tsukuba512_L.png",
        //"../../res/tsukuba512_R.png",
        "lt.png", "rt.png",
        "tsukuba_medianbirchfieldwta.png", &bf, &wta, true);   
}

