#include <gtest/gtest.h>
#include <tdvision/ssddev.hpp>
#include <tdvision/dynamicprogdev.hpp>
#include <tdvision/semiglobaloptcpu.hpp>
#include <tdvision/semiglobaldev.hpp>
#include <tdvision/birchfieldcostdev.hpp>
#include "stereotest.hpp"

#if 1
TEST(TestSemiGlobalDev, WithSSD)
{
    tdv::SSDDev ssd(64);
    tdv::SemiGlobalDev sg;

   runStereoTest(
#if 0
       "../../res/tsukuba512_L.png",
       "../../res/tsukuba512_R.png",
       "tsukuba_ssdsgdev.png",
#else
       "q_left.png",
       "q_right.png",
       "q_ssdsgdev.png",
#endif
       &ssd, &sg, true);

}
#endif

#if 0
TEST(TestSemiGlobalCPU, WithSSD)
{
    tdv::SSDDev ssd(16);
    tdv::SemiGlobalOptCPU sg;
    
   runStereoTest(
       "../../res/tsukuba512_L.png",
       "../../res/tsukuba512_R.png",
       "tsukuba_ssdsgcpu.png",
       &ssd, &sg, true);

}
#endif
