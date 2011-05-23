#include <tdvision/tdvision.hpp>
#include <gtest/gtest.h>
#include "errorhandler.hpp"

TEST(CaptureTest, InfiniteLoopToCheckMemLeak)
{
    tdv::CaptureProc cap;
    tdv::TWorkUnitProcess<tdv::CvMatSink> sink;
    tdv::ArrayProcessGroup  grp;
    
    cap.init(0);
    
    sink.input(cap.output());
    
    grp.addProcess(&cap);
    grp.addProcess(&sink);
    
    ErrorHandler errHdl;
    tdv::ProcessRunner runner(grp, &errHdl);
    runner.run();
    runner.join();
}
