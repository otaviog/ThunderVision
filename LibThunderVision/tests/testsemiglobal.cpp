#include <gtest/gtest.h>
#include <tdvision/ssddev.hpp>
#include <tdvision/dynamicprogdev.hpp>
#include <tdvision/semiglobaloptcpu.hpp>
#include <tdvision/birchfieldcostdev.hpp>
#include "stereotest.hpp"

TEST(TestSemiGlobal, WithSSD)
{
    //tdv::SSDDev ssd(128);
    tdv::BirchfieldCostDev ssd(128);
    tdv::SemiGlobalOptCPU sg;
    
   runStereoTest(
       "../../res/tsukuba512_L.png",
       "../../res/tsukuba512_R.png",
       "tsukuba_ssdsg.png",
       &ssd, &sg);

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

#endif
