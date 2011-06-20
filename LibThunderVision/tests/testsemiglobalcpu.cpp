#include <gtest/gtest.h>
#include <tdvision/ssddev.hpp>
#include <tdvision/dynamicprogdev.hpp>
#include <tdvision/semiglobalcpu.hpp>
#include <tdvision/wtacpu.hpp>
#include "stereotest.hpp"

TEST(TestSemiGlobalCPU, WithSSD)
{
    tdv::SSDDev cost(256);
    tdv::SemiGlobalCPU sg;
    
   runStereoTest(
#if 0
       "../../res/tsukuba512_L.png",
       "../../res/tsukuba512_R.png",
#else
       "q_left.png",
       "q_right.png",
#endif
       "tsukuba_ssdsgcpu.png",
       &cost, &sg, true);
}
