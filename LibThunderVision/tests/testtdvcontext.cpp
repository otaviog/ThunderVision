#include <gtest/gtest.h>
#include <tdvision/tdvcontext.hpp>
#include <tdvision/camerastereoinputsource.hpp>
#include <cv.h>
#include <highgui.h>

using namespace tdv;

TEST(TestTDVContext, CameraView)
{
    TDVContext ctx;
    CameraStereoInputSource inputSrc;
    
    //ctx.loadSpecFromFile("");            
    ctx.start(&inputSrc);
    
    ReadPipe<IplImage*> *lftImgP, *rgtImgP;
    ctx.dupInputSource(&lftImgP, &rgtImgP);
    
    //ctx.runReconstruction();
    //ctx.releaseReconstruction();
    
    //ctx.runCalibration();
    //ctx.releaseCalibration();
    
    IplImage *lftImg, *rgtImg;
    while ( lftImgP->read(&lftImg) && rgtImgP->read(&rgtImg) )
    {
        cvShowImage("Left", lftImg);
        cvShowImage("Right", rgtImg);        
        cvWaitKey(0);
    }
}

TEST(TestTDVContext, ShouldCreateReconstruction)
{
    TDVContext ctx;
    CameraStereoInputSource inputSrc;
    
    ctx.start(&inputSrc);
    Reconstruction *reconst = ctx.runReconstruction("CPU");
    
    reconst->start();
    reconst->stop();
    
    
}
