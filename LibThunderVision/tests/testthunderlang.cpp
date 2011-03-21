#include <gtest/gtest.h>
#include <tdvision/thunderlang.hpp>

TEST(ThunderLangTest, ParseFile)
{
    tdv::ThunderSpec spec;
    tdv::ThunderLangParser parser(spec);
    
    parser.parseFile("../../res/camerasdesc.tl");
    
    tdv::CamerasDesc desc = spec.camerasDesc("default");
    EXPECT_DOUBLE_EQ(1.0, desc.leftCamera().distortion()[0]);
    EXPECT_DOUBLE_EQ(2.0, desc.leftCamera().distortion()[1]);
    EXPECT_DOUBLE_EQ(3.0, desc.leftCamera().distortion()[2]);
    EXPECT_DOUBLE_EQ(4.0, desc.leftCamera().distortion()[3]);
    EXPECT_DOUBLE_EQ(5.0, desc.leftCamera().distortion()[4]);
    
    EXPECT_DOUBLE_EQ(1.0, desc.leftCamera().intrinsics()[0]);
    EXPECT_DOUBLE_EQ(2.0, desc.leftCamera().intrinsics()[1]);
    EXPECT_DOUBLE_EQ(3.0, desc.leftCamera().intrinsics()[2]);
    EXPECT_DOUBLE_EQ(4.0, desc.leftCamera().intrinsics()[3]);
    EXPECT_DOUBLE_EQ(5.0, desc.leftCamera().intrinsics()[4]);
    EXPECT_DOUBLE_EQ(6.0, desc.leftCamera().intrinsics()[5]);
    EXPECT_DOUBLE_EQ(7.0, desc.leftCamera().intrinsics()[6]);
    EXPECT_DOUBLE_EQ(8.0, desc.leftCamera().intrinsics()[7]);
    EXPECT_DOUBLE_EQ(9.0, desc.leftCamera().intrinsics()[8]);    
}

TEST(ThunderLangTest, WriteFile)
{
    tdv::ThunderSpec spec;
    tdv::ThunderLangParser parser(spec);
    
    parser.parseFile("../../res/camerasdesc.tl");
    
    tdv::ThunderLangWriter writer;
    writer.write("camerasdesc-w.tl", spec);
    
    parser.parseFile("camerasdesc-w.tl");
}

