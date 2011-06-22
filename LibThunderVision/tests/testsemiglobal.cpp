#include <gtest/gtest.h>
#include <tdvision/ssddev.hpp>
#include <tdvision/semiglobaldev.hpp>
#include <tdvision/birchfieldcostdev.hpp>
#include "stereotest.hpp"

#if 0
TEST(TestSemiGlobalDev, WithBT)
{    
    tdv::BirchfieldCostDev cost(64);
    tdv::SemiGlobalDev sg;

   runStereoTest(
       "../../res/tsukuba512_L.png",
       "../../res/tsukuba512_R.png",
       "tsukuba_btsgdev.png",
       &cost, &sg, false, false);
}
#endif


TEST(TestSemiGlobalDev, WithSSD)
{        
    tdv::SSDDev cost(128);
    tdv::SemiGlobalDev sg;

    runStereoTest(
       "../../res/tsukuba512_L.png",
       "../../res/tsukuba512_R.png",

        "tsukuba_ssdsgdev.png",
        &cost, &sg, true, false);
}

