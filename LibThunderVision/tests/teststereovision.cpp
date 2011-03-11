#include <gtest/gtest.hpp>

TEST(TestStereoVision, Rationalle)
{
    StereoMactherProcess svgen;
    
    
    svgen.addPreFilter();
    svgen.addPosFilter();
    svgen.
    svgen.costFunction();
    svgen.optimizer();
    
    svgen.process();
    
    svgen.input(imgl, imgr);
    FloatImage img = svgen.output();
    
    ReconstructionProcess recproc;
    GeometryGenerator gen;
    
    recproc.setGeometryGenerator();    
    
    StereoVisionDescription desc;
    StereoMatcherProcess proc = desc.createMatcherProcess();
    
    
}
