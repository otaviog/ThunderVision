#include <gtest/gtest.h>
#include <tdvision/thunderlang.hpp>

static void testSampleDesc1(const tdv::CamerasDesc &desc)
{
    EXPECT_DOUBLE_EQ(1, desc.leftCamera().distortion()[0]);
    EXPECT_DOUBLE_EQ(2.0, desc.leftCamera().distortion()[1]);
    EXPECT_DOUBLE_EQ(3.3e14, desc.leftCamera().distortion()[2]);
    EXPECT_DOUBLE_EQ(4.1e-10, desc.leftCamera().distortion()[3]);
    EXPECT_DOUBLE_EQ(-5.3e10, desc.leftCamera().distortion()[4]);
    
    EXPECT_DOUBLE_EQ(1.0, desc.leftCamera().intrinsics()[0]);
    EXPECT_DOUBLE_EQ(2.5, desc.leftCamera().intrinsics()[1]);
    EXPECT_DOUBLE_EQ(3.66, desc.leftCamera().intrinsics()[2]);
    EXPECT_DOUBLE_EQ(4.0, desc.leftCamera().intrinsics()[3]);
    EXPECT_DOUBLE_EQ(-5.4, desc.leftCamera().intrinsics()[4]);
    EXPECT_DOUBLE_EQ(6.0, desc.leftCamera().intrinsics()[5]);
    EXPECT_DOUBLE_EQ(7.0, desc.leftCamera().intrinsics()[6]);
    EXPECT_DOUBLE_EQ(8.0, desc.leftCamera().intrinsics()[7]);
    EXPECT_DOUBLE_EQ(9.9, desc.leftCamera().intrinsics()[8]);    
    
    EXPECT_TRUE(desc.hasFundamentalMatrix());
    EXPECT_TRUE(desc.hasExtrinsics());
}

TEST(ThunderLangTest, ParseFile)
{
    tdv::ThunderSpec spec;
    tdv::ThunderLangParser parser(spec);
    
    parser.parseFile("../../res/camerasdesc.tl");
    
    tdv::CamerasDesc desc = spec.camerasDesc("default");
    testSampleDesc1(desc);
}

TEST(ThunderLangTest, WriteFile)
{
    tdv::ThunderSpec spec;
    tdv::ThunderLangParser parser(spec);
    
    parser.parseFile("../../res/camerasdesc.tl");
    
    tdv::ThunderLangWriter writer;
    writer.write("camerasdesc-w.tl", spec);    
    parser.parseFile("camerasdesc-w.tl");
    
    tdv::ThunderSpec spec2;
    tdv::ThunderLangParser parser2(spec2);
    
    parser2.parseFile("../../res/camerasdesc.tl");
    testSampleDesc1(spec2.camerasDesc("default"));
}

