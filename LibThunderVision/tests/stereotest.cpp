#include <tdvision/imagereader.hpp>
#include <tdvision/imagewriter.hpp>
#include <tdvision/floatconv.hpp>
#include <tdvision/rgbconv.hpp>
#include <tdvision/medianfilterdev.hpp>
#include <tdvision/medianfiltercpu.hpp>
#include "stereotest.hpp"

void runStereoTest(
    const std::string &leftImg,
    const std::string &rightImg,
    const std::string &outputImg, 
    tdv::MatchingCost *mcost, 
    tdv::Optimizer *opt,
    bool medianFilter)
{
    tdv::ImageReader readerL(leftImg);
    tdv::ImageReader readerR(rightImg);
    
    tdv::FloatConv fconvl, fconvr;
    tdv::RGBConv rconv;
    
    tdv::ImageWriter writer(outputImg);    
    tdv::MedianFilterCPU mdFilterL, mdFilterR;
    
    fconvl.input(readerL.output());
    fconvr.input(readerR.output());
    
    if ( medianFilter )
    {
        mdFilterL.input(fconvl.output());
        mdFilterR.input(fconvr.output());
        mcost->inputs(mdFilterL.output(), mdFilterR.output());
    }
    else
    {
        mcost->inputs(fconvl.output(), fconvr.output());    
    }   
    
    opt->input(mcost->output());    
    rconv.input(opt->output());
    writer.input(rconv.output());
    
    readerL.update();
    readerR.update();
    fconvl.update();
    fconvr.update();
    
    if ( medianFilter )
    {
        mdFilterL.update();
        mdFilterR.update();
    }
    
    mcost->update();
    opt->update();
    rconv.update();
    writer.update();    
}
