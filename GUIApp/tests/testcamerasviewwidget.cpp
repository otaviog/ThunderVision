#include <tdvbasic/log.hpp>
#include <QApplication>
#include <QPushButton>
#include <camerasviewwidget.hpp>
#include <tdvision/pipe.hpp>
#include <tdvision/captureproc.hpp>
#include <tdvision/imagesink.hpp>
#include <tdvision/processrunner.hpp>
#include <tdvision/workunitprocess.hpp>
#include <tdvision/exceptionreport.hpp>

class ErrorHandle: public tdv::ExceptionReport
{
public:    
    void errorOcurred(const std::exception &err)
    {
        std::cout<<err.what()<<std::endl;        
    }
    
private:
    CameraWidget *w0, *w1;
};

int main(int argc, char *argv[])
{
    tdv::TdvGlobalLogDefaultOutputs();
    
    QApplication app(argc, argv);
    
    tdv::CaptureProc capture0(0);
    tdv::CaptureProc capture1(1);
    tdv::ImageSink sink0, sink1;
                   
    sink0.input(capture0.output());
    sink1.input(capture1.output());
    
    CamerasViewWidget *wid = 
        new CamerasViewWidget(capture0.colorImage(), capture1.colorImage());
    
    QPushButton *pb = new QPushButton("Photochar");
    pb->show();
    
    QObject::connect(pb, SIGNAL(clicked()), 
                     wid, SLOT(photoChar()));
    
    tdv::WorkUnitProcess s0proc(sink0);
    tdv::WorkUnitProcess s1proc(sink1);
    
    tdv::Process *wus[] = { 
        &capture0, &capture1, &s0proc, &s1proc, 
        NULL
    };
    
    ErrorHandle errHdl;
    tdv::ProcessRunner runner(wus, &errHdl);
    runner.run();
    wid->start();

    wid->show();
    int r = app.exec();    
    
    capture0.finish();
    capture1.finish();
    
    runner.join();
    
    return r;
}
