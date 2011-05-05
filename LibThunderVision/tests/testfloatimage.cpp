#include <gtest/gtest.h>
#include <tdvision/floatimage.hpp>
#include <tdvision/process.hpp>
#include <tdvision/pipe.hpp>
#include "errorhandler.hpp"

#define BUFF_SIZE 10
#define PROD_QTY 1000

class Prod: public tdv::Process
{
public:
    void process()
    {
        for (int i=0; i<PROD_QTY; i++)
        {            
            p->write(tdv::FloatImage::CreateCPU(tdv::Dim(256, 256)));
        }
        
        p->finish();
    }
    
    tdv::ReadWritePipe<tdv::FloatImage> *p;
};

class Consu: public tdv::Process
{
public:
    void process()
    {
        tdv::FloatImage img;
        while ( p->read(&img) )
        {
            usleep(1000);
        }        
    }
    
    tdv::ReadWritePipe<tdv::FloatImage> *p;
};

TEST(TestFloatImage, Assigment)
{
    tdv::FloatImage image = tdv::FloatImage::CreateCPU(tdv::Dim(256, 256));
    tdv::FloatImage img2 = image;
}

TEST(TestFloatImage, Pipe)
{
    tdv::ReadWritePipe<tdv::FloatImage> p(BUFF_SIZE);
    
    Prod pr;
    Consu cs;
    pr.p = &p;
    cs.p = &p;
    
    ErrorHandler errHdl;
    tdv::ArrayProcessGroup grp;
    grp.addProcess(&pr);
    grp.addProcess(&cs);
    
    tdv::ProcessRunner runner(grp, &errHdl);
    runner.run();
    
    runner.join();
}
