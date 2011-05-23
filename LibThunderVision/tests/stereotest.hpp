#ifndef TDV_STEREOTEST_HPP
#define TDV_STEREOTEST_HPP

#include <string>
#include <tdvision/matchingcost.hpp>
#include <tdvision/optimizer.hpp>

void runStereoTest(
    const std::string &leftImg,
    const std::string &rightImg,
    const std::string &outputImg, 
    tdv::MatchingCost *mcost, 
    tdv::Optimizer *opt,
    bool medianFilter = false);


#endif /* TDV_STEREOTEST_HPP */
