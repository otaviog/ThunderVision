#include <gtest/gtest.h>
#include <tdvision/imagereader.hpp>
#include <tdvision/calibration.hpp>
#include <tdvision/rectificationcv.hpp>
#include <tdvision/thunderlang.hpp>
#include <highgui.h>

TEST(TestCalibration, Calibration)
{
    tdv::ImageReader readerL("../../res/OpenCVBook_StereoDataL/", tdv::ImageReader::Directory);
    tdv::ImageReader readerR("../../res/OpenCVBook_StereoDataR/", tdv::ImageReader::Directory);
    
    tdv::Calibration calib(13);
    calib.input(readerL.output(), readerR.output());
    calib.chessPattern(tdv::ChessboardPattern(tdv::Dim(9, 6)));
    for (size_t i=0; i<14; i++)
    {
        readerL.update();
        readerR.update();
        calib.update();
    }

// p M1
// $1 = {{534.73022340234274, 0, 335.14036440321371}, {0, 534.73022340234274, 240.20358200823657}, {0, 
//     0, 1}}
// (gdb) p D1
// $2 = {-0.27548644140618689, -0.0074741447321146201, 0, 0, 0.19877104747410226}
// (gdb) p M2
// $3 = {{534.73022340234274, 0, 334.01685399468971}, {0, 534.73022340234274, 241.57913691145697}, {0, 
//     0, 1}}
// (gdb) p D2
// $4 = {-0.28088283603968645, 0.093133149965471215, 0, 0, -0.014268852873648923}
// (gdb) p T
// $5 = {-3.338521775036563, 0.04875726580427088, -0.10610701997669005}
// (gdb) p E
// $6 = {{0.00052298343399445103, 0.10522183916666465, 0.05063656492846199}, {-0.034564465155821761, 
//     -0.059921868662649605, 3.3394911303200199}, {-0.032337732207657177, -3.3382054064188389, 
//     -0.058684316420089928}}
// (gdb) p F
// $7 = {{-2.5604703173677883e-08, -5.1515474183824866e-06, -7.9654502412560401e-05}, {
//     1.6922388227714652e-06, 2.9337098672527604e-06, -0.088699079381738072}, {0.00044633859910145731, 
//     0.088405574779886281, 1}}

    tdv::CamerasDesc cdesc(calib.camerasDesc()); 

    EXPECT_EQ(-0.27548644140618689, cdesc.leftCamera().distortion()[0]);
    
    tdv::RectificationCV rect;
    
    rect.camerasDesc(cdesc);
    
    rect.leftImgInput(readerL.output());
    rect.rightImgInput(readerR.output());
    
    readerL.reset();
    readerR.reset();
    
    readerL.update();
    readerR.update();
    rect.update();
    
    tdv::FloatImage limg, rimg;
    
    ASSERT_TRUE(rect.leftImgOutput()->read(&limg));
    ASSERT_TRUE(rect.rightImgOutput()->read(&rimg));
        
    cvShowImage("L", limg.cpuMem());
    cvShowImage("R", rimg.cpuMem());
    
    cvWaitKey(0);
    
    tdv::ThunderSpec spec;
    spec.camerasDesc("default") = cdesc;
    tdv::ThunderLangWriter wrt;
    wrt.write("calib.tl", spec);
}
