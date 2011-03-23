#include <gtest/gtest.h>
#include <tdvision/rectificationcv.hpp>
#include <tdvision/imagereader.hpp>
#include <tdvision/imagewriter.hpp>
#include <tdvision/thunderlang.hpp>

class TestRectification : public ::testing::Test
{
protected:
    static void SetUpTestCase()
    {
        tdv::ThunderSpec extSpec;
        tdv::ThunderSpec insSpec;
                
        {
            tdv::ThunderLangParser parser(extSpec);
            parser.parseFile("../../res/calib_ext.tl");
        }
        
        {
            tdv::ThunderLangParser parser(insSpec);
            parser.parseFile("../../res/calib_ins.tl");
        }
        
        extCalib = extSpec.camerasDesc("default");
        insCalib = insSpec.camerasDesc("default");
    }
    
    static tdv::CamerasDesc extCalib;
    static tdv::CamerasDesc insCalib;
};

tdv::CamerasDesc TestRectification::extCalib;
tdv::CamerasDesc TestRectification::insCalib;

TEST_F(TestRectification, UncalibratedRect)
{
    tdv::ImageReader readerL("../../res/rect-left.png");
    tdv::ImageReader readerR("../../res/rect-right.png");
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
    
TEST_F(TestRectification, InstrinsicRect)
{
    tdv::ImageReader readerL("../../res/rect-calib-left.png");
    tdv::ImageReader readerR("../../res/rect-calib-right.png");
    tdv::ImageWriter writerL("rect-left-done.png");
    tdv::ImageWriter writerR("rect-right-done.png");

    tdv::RectificationCV rectCV;            
    rectCV.camerasDesc(insCalib);
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

TEST_F(TestRectification, ExtrinsicRect)
{
    tdv::ImageReader readerL("../../res/rect-calib-left.png");
    tdv::ImageReader readerR("../../res/rect-calib-right.png");
    tdv::ImageWriter writerL("rect-calib-left-done.png");
    tdv::ImageWriter writerR("rect-calib-right-done.png");

    tdv::RectificationCV rectCV;         
    rectCV.camerasDesc(extCalib);    
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
