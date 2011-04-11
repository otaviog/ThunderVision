#include <gtest/gtest.h>
#include <tdvision/tdvision.hpp>

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

static void runTest(
    const std::string &lftInput, const std::string &rgtInput, 
    const std::string &lftOutput, const std::string &rgtOutput, 
    tdv::RectificationCV &rect)
{
    tdv::ImageReader readerL(lftInput);
    tdv::ImageReader readerR(rgtInput);    
    tdv::RGBConv rconvL, rconvR;
    tdv::ImageWriter writerL(lftOutput);
    tdv::ImageWriter writerR(rgtOutput);
        
    rect.leftImgInput(readerL.output());
    rect.rightImgInput(readerR.output());
    
    rconvL.input(rect.leftImgOutput());
    rconvR.input(rect.rightImgOutput());
    
    writerL.input(rconvL.output());
    writerR.input(rconvR.output());
    
    readerL.update();
    readerR.update();
    rect.update();
    rconvL.update();
    rconvR.update();
    writerL.update();
    writerR.update();
}

TEST_F(TestRectification, UncalibratedRect)
{
    tdv::RectificationCV rectCV; 
    
    runTest("../../res/rect-left.png",
            "../../res/rect-right.png",
            "rect-left-done.png",
            "rect-right-done.png",
        rectCV);
}
    
TEST_F(TestRectification, InstrinsicRect)
{
    tdv::RectificationCV rectCV;            
    rectCV.camerasDesc(insCalib);
    runTest("../../res/rect-calib-left.png",
            "../../res/rect-calib-right.png",
            "rect-icalib-left-done.png",
            "rect-icalib-right-done.png",
        rectCV);
}

TEST_F(TestRectification, ExtrinsicRect)
{
    tdv::RectificationCV rectCV;         
    rectCV.camerasDesc(extCalib);    

    runTest("../../res/rect-calib-left.png",
            "../../res/rect-calib-right.png",
            "rect-ecalib-left-done.png",
            "rect-ecalib-right-done.png",
        rectCV);

}
