#include <gtest/gtest.h>
#include <tdvision/ssddev.hpp>
#include <tdvision/semiglobaldev.hpp>
#include <tdvision/birchfieldcostdev.hpp>
#include "stereotest.hpp"

TEST(TestSemiGlobalDev, WithBT)
{    
    tdv::BirchfieldCostDev cost(256);
    tdv::SemiGlobalDev sg;

   runStereoTest(
       "../../res/tsukuba512_L.png",
       "../../res/tsukuba512_R.png",
       "tsukuba_btsgdev.png",
       &cost, &sg, true, false);
}

TEST(TestSemiGlobalDev, WithSSD)
{        
    tdv::SSDDev cost(256);
    tdv::SemiGlobalDev sg;

    runStereoTest(
       "../../res/tsukuba512_L.png",
       "../../res/tsukuba512_R.png",

        "tsukuba_ssdsgdev.png",
        &cost, &sg, true, false);
}
