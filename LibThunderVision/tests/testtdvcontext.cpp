#include <gtest/gtest.h>
#include <tdvision/tdvcontext.hpp>
#include <tdvision/capturestereoinputsource.hpp>
#include <tdvision/reconstruction.hpp>
#include <cv.h>
#include <highgui.h>

using namespace tdv;

TEST(TestTDVContext, CameraView)
{
    TDVContext ctx;
    CaptureStereoInputSource inputSrc;
    inputSrc.init("../../res/cam0.avi", "../../res/cam1.avi");
        
    ctx.start(&inputSrc);
    
    ReadPipe<CvMat*> *lftImgP, *rgtImgP;
    ctx.dupInputSource(&lftImgP, &rgtImgP);        
    
    CvMat *lftImg, *rgtImg;
    EXPECT_TRUE(lftImgP->read(&lftImg));
    EXPECT_TRUE(rgtImgP->read(&rgtImg));
    
    ctx.dispose();
}

TEST(TestTDVContext, ShouldCreateReconstruction)
{
    TDVContext ctx;
    
    CaptureStereoInputSource inputSrc;
    inputSrc.init("../../res/cam0.avi", "../../res/cam1.avi");

    ctx.start(&inputSrc);
    Reconstruction *reconst = ctx.runReconstruction("CPU");
    
    ASSERT_TRUE(reconst != NULL);

    reconst->step();
    reconst->pause();        
    
    ctx.releaseReconstruction(reconst);
    
    reconst = ctx.runReconstruction("Device");
    reconst->step();
    
    ctx.releaseReconstruction(reconst);
    ctx.dispose();
}
