#include <tdvbasic/log.hpp>
#include <QApplication>
#include <camerawidget.hpp>
#include <tdvision/pipe.hpp>
#include <tdvision/captureproc.hpp>
#include <tdvision/imagesink.hpp>
#include <tdvision/processrunner.hpp>
#include <tdvision/cudaprocess.hpp>
#include <tdvision/workunitprocess.hpp>
#include <tdvision/rgbconv.hpp>
#include <tdvision/medianfiltercpu.hpp>
#include <tdvision/medianfilterdev.hpp>
#include <tdvision/imageresize.hpp>

class ErrorHandle: public tdv::ProcessExceptionReport
{
public:
    ErrorHandle(CameraWidget *wid0, CameraWidget *wid1)
        : w0(wid0), w1(wid1)
    {
    }
    
    void errorOcurred(const std::exception &err)
    {
        std::cout<<err.what()<<std::endl;
        w0->close();
        w1->close();
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
    tdv::RGBConv conv1;    
    tdv::ImageSink sink0;
    tdv::MedianFilterDev median;
    tdv::ImageResize resize(tdv::Dim(512, 512));
    
    // First Camera
    CameraWidget *wid0 = new CameraWidget(capture0.colorImage(), false);
    sink0.input(capture0.output());
    
    // Second Camera
    resize.input(capture1.output());
    median.input(resize.output());
    conv1.input(median.output());    
    
    CameraWidget *wid1 = new CameraWidget(conv1.output(), true); 
    
    wid0->show();
    wid1->show();

    tdv::CUDAProcess cproc(0);
    cproc.addWork(&median);
    cproc.addWork(&conv1);
    tdv::WorkUnitProcess rproc(resize);
    tdv::WorkUnitProcess sinkproc(sink0);
    tdv::Process *wus[] = { 
        &capture0, wid0, &sinkproc, 
        &capture1, &rproc, &cproc, wid1, NULL
    };
    
    ErrorHandle errHdl(wid0, wid1);
    tdv::ProcessRunner runner(wus, &errHdl);
    runner.run();
    
    int r = app.exec();    
    
    capture0.finish();
    capture1.finish();
    
    runner.join();
    
    return r;
}
