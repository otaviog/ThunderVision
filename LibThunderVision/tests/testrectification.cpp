#include <gtest/gtest.h>
#include <tdvision/rectificationcv.hpp>
#include <tdvision/imagereader.hpp>
#include <tdvision/imagewriter.hpp>

TEST(TestRectification, RectificationCV)
{
    tdv::ImageReader readerL("../../res/rect-left2.png");
    tdv::ImageReader readerR("../../res/rect-right2.png");
    tdv::ImageWriter writerL("rect-left-done.png");
    tdv::ImageWriter writerR("rect-right-done.png");

    tdv::RectificationCV rectCV;            
    rectCV.leftImgInput(readerL.output());
    rectCV.rightImgInput(readerR.output());
    
    writerL.input(rectCV.leftImgOutput());
    writerR.input(rectCV.rightImgOutput());
    
    readerL.update();
    readerR.update();
    rectCV.update();

    writerL.update();
    writerR.update();
}
    
