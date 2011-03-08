#include <tdvbasic/log.hpp>
#include <QApplication>
#include <camerawidget.hpp>
#include <tdvision/pipe.hpp>
#include <tdvision/captureproc.hpp>
#include <tdvision/imagesink.hpp>
#include <tdvision/processrunner.hpp>
#include <tdvision/cudaprocess.hpp>
#include <tdvision/workunitprocess.hpp>

class ErrorHandle: public tdv::ProcessExceptionReport
{
public:
    ErrorHandle(CameraWidget *wid0)
        : w0(wid0)
    {
    }
    
    void errorOcurred(const std::exception &err)
    {
        std::cout<<err.what()<<std::endl;
        w0->close();
    }
    
private:
    CameraWidget *w0;
};

int main(int argc, char *argv[])
{
    tdv::TdvGlobalLogDefaultOutputs();
    
    QApplication app(argc, argv);
    
    tdv::CaptureProc capture0(0);    
    tdv::ImageSink sink0;
    
    CameraWidget *wid0 = new CameraWidget(capture0.colorImage(), false);
    sink0.input(capture0.output());
    
    wid0->show();
    
    tdv::WorkUnitProcess sink0proc(sink0);
    
    tdv::Process *procs[] = { 
        &capture0, wid0, &sink0proc, NULL
    };
    
    ErrorHandle errHdl(wid0);
    tdv::ProcessRunner runner(procs, &errHdl);
    runner.run();
    
    int r = app.exec();    
    
    capture0.finish();    
    runner.join();
    
    return r;
}
