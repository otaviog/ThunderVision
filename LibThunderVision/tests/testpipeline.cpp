#include <gtest/gtest.h>
#include <tdvision/pipe.hpp>
#include <tdvision/workunit.hpp>
#include <tdvision/workunitrunner.hpp>

static void procSimple(tdv::ReadPipe<int> *inp, tdv::WritePipe<int> *outp)
{
    int a = inp->read();
    EXPECT_EQ(2, a);
    outp->write(a + 2);
}

static void procConv(tdv::ReadPipe<int> *inp, tdv::WritePipe<float> *outp)
{
    int a = inp->read();
    outp->write(float(a) + 2.0);
}

TEST(PipeTest, Connections)
{
    tdv::ReadWritePipe<int, int, tdv::PasstruPipeAdapter<int> > pipe;
    pipe.write(2);
    procSimple(&pipe, &pipe);
    EXPECT_EQ(4, pipe.read());
    
    pipe.write(4);
    
    tdv::ReadWritePipe<int, float, tdv::CastPipeAdapter<int, float> > pipeConv;
    procConv(&pipe, &pipeConv);
    EXPECT_FLOAT_EQ(6.0, pipeConv.read());    
}

class Filter1: public tdv::WorkUnit
{
public:
    Filter1(tdv::ReadPipe<int> *i,
            tdv::WritePipe<int> *o)
        : WorkUnit("Filter 1")
    {
        inpipe = i;
        outpipe = o;
    }

    void process()
    {
        while ( inpipe->waitPacket() )
        {
            int value = inpipe->read();
            outpipe->write(value*2);
        }
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
        : WorkUnit("Filter 2")
    {
        inpipe = i;
        outpipe = o;
    }
    
    void process()
    {
        while ( inpipe->waitPacket() )
        {
            int value = inpipe->read();
            outpipe->write(value*2);
        }
    }    

    const char* name() const
    {
        return "Filter2";
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
    
    tdv::WorkUnitRunner runner;
    tdv::WorkUnit *procs[] = { &f1, &f2 };
    runner.run(procs, 2);
        
    p1.write(2);
    p1.write(3);
    p1.write(4);
    
    p3.waitPacket();
    EXPECT_FLOAT_EQ(8.0f, p3.read());
    p3.waitPacket();
    EXPECT_FLOAT_EQ(12.0f, p3.read());
    p3.waitPacket();
    EXPECT_FLOAT_EQ(16.0f, p3.read());
    
    p1.end();
    p2.end();
    p3.end();
}
