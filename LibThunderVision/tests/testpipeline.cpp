#include <gtest/gtest.h>
#include <tdvision/pipe.hpp>
#include <tdvision/workunit.hpp>
#include <tdvision/processrunner.hpp>
#include <tdvision/workunitprocess.hpp>
#include <tdvision/processgroup.hpp>
#include "errorhandler.hpp"

static void procSimple(tdv::ReadPipe<int> *inp, tdv::WritePipe<int> *outp)
{
    int a;
    inp->read(&a);
    EXPECT_EQ(2, a);
    outp->write(a + 2);
}

static void procConv(tdv::ReadPipe<int> *inp, tdv::WritePipe<int> *outp)
{
    int a;
    inp->read(&a);
    outp->write(a + 2);
}

TEST(PipeTest, Connections)
{
    tdv::ReadWritePipe<int, int, tdv::PasstruPipeAdapter<int> > pipe;
    pipe.write(2);
    procSimple(&pipe, &pipe);
    int v;
    EXPECT_TRUE(pipe.read(&v));
    EXPECT_EQ(4, v);
    
    pipe.write(4);
    
    tdv::ReadWritePipe<float, int, 
        tdv::CastPipeAdapter<float, int> > pipeConv;
    procConv(&pipe, &pipeConv);
    float vf;
    EXPECT_TRUE(pipeConv.read(&vf));
    EXPECT_FLOAT_EQ(6.0, vf);    
}

class Filter1: public tdv::WorkUnit
{
public:
    Filter1(tdv::ReadPipe<int> *i,
            tdv::WritePipe<int> *o)
    {
        workName("Filter 1");
        inpipe = i;
        outpipe = o;
    }

    bool update()
    {
        workName("Filter 2");
        
        int value;
        while ( inpipe->read(&value) )
        {
            outpipe->write(value*2);
        }
        
        return false;
    }
       
private:
    tdv::ReadPipe<int> *inpipe;
    tdv::WritePipe<int> *outpipe;
};

class Filter2: public tdv::WorkUnit
{
public:    
    Filter2(tdv::ReadPipe<int> *i,
            tdv::WritePipe<float> *o)
    {
        workName("Filter 2");
        inpipe = i;
        outpipe = o;
    }
    
    bool update()
    {
        int value;
        while ( inpipe->read(&value) )
        {
            outpipe->write(value*2);
        }
        
        return false;
    }    
    
private:
    tdv::ReadPipe<int> *inpipe;
    tdv::WritePipe<float> *outpipe;

};

TEST(PipeTest, PipeAndFilter)
{
    tdv::ReadWritePipe<int, int> p1, p2;
    tdv::ReadWritePipe<float, float> p3;
    
    Filter1 f1(&p1, &p2);
    Filter2 f2(&p2, &p3);
    tdv::WorkUnitProcess pr1(f1);
    tdv::WorkUnitProcess pr2(f2);
    
    ErrorHandler errHdl;
    tdv::Process *procs[] = { &pr1, &pr2, NULL};    
    tdv::ArrayProcessGroup grp(procs);
    tdv::ProcessRunner runner(grp, &errHdl);
    runner.run();
        
    p1.write(2);
    p1.write(3);
    p1.write(4);    

    float vf;
    p3.read(&vf);
    EXPECT_FLOAT_EQ(8.0f, vf);
    p3.read(&vf);
    EXPECT_FLOAT_EQ(12.0f, vf);
    p3.read(&vf);
    EXPECT_FLOAT_EQ(16.0f, vf);
    
    p1.finish();
    p2.finish();
    p3.finish();

    runner.join();     
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
   
  return RUN_ALL_TESTS();
}

