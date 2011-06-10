#include <gtest/gtest.h>
#include <tdvision/ssddev.hpp>
#include <tdvision/dynamicprogdev.hpp>
#include <tdvision/semiglobalcpu.hpp>
#include <tdvision/semiglobaldev.hpp>
#include <tdvision/birchfieldcostdev.hpp>
#include "stereotest.hpp"

TEST(TestSemiGlobalDev, WithSSD)
{    
    tdv::BirchfieldCostDev cost(64);
    //tdv::SSDDev cost(256);
    tdv::SemiGlobalDev sg;

   runStereoTest(
       "../../res/tsukuba_L.png",
       "../../res/tsukuba_R.png",
       "tsukuba_btsgdev.png",
       &cost, &sg, true, true);
}

#if 0
TEST(TestSemiGlobalCPU, WithSSD)
{
    tdv::SSDDev cost(16);
    tdv::SemiGlobalOptCPU sg;
    
   runStereoTest(
       "../../res/tsukuba512_L.png",
       "../../res/tsukuba512_R.png",
       "tsukuba_ssdsgcpu.png",
       &cost, &sg, true);

}
#endif
