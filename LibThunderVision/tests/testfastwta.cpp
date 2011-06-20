#include <gtest/gtest.h>
#include <tdvision/fastwtamatcher.hpp>
#include <tdvision/imagereader.hpp>
#include <tdvision/imagewriter.hpp>
#include <tdvision/floatconv.hpp>
#include <tdvision/rgbconv.hpp>

TEST(TestFastWTA, Run)
{
    tdv::FastWTAMatcher matcher(32);
    
    tdv::ImageReader readerL("../../res/tsukuba512_L.png");
    tdv::ImageReader readerR("../../res/tsukuba512_R.png");
    tdv::ImageWriter writer("fastwta.png");
    
    tdv::FloatConv fconvl, fconvr;
    tdv::RGBConv rconv;
    fconvl.input(readerL.output());
    fconvr.input(readerR.output());
    
    matcher.inputs(fconvl.output(), fconvr.output());
    
    rconv.input(matcher.output());
    
    writer.input(rconv.output());
    
    readerL.update();
    readerR.update();
    fconvl.update();
    fconvr.update();
    
    matcher.update();
    
    rconv.update();
    writer.update();
}
