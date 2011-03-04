#include <tdvbasic/log.hpp>
#include <QApplication>
#include <camerawidget.hpp>
#include <tdvision/pipe.hpp>
#include <tdvision/capturewu.hpp>
#include <tdvision/imagesink.hpp>
#include <tdvision/workunitrunner.hpp>
#include <tdvision/rgbconv.hpp>
#include <tdvision/medianfilterwucpu.hpp>
#include <tdvision/medianfilterwudev.hpp>
#include <tdvision/resizeimagewu.hpp>

class ErrorHandle: public tdv::WorkExceptionReport
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
    
    tdv::CaptureWU capture0(0);
    tdv::CaptureWU capture1(1);
    tdv::RGBConv conv1;    
    tdv::ImageSink sink0;
    tdv::MedianFilterWUDev median;
    tdv::ResizeImageWU resize(tdv::Dim(512, 512));
    
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

    tdv::WorkUnit *wus[] = { &capture0, wid0, &sink0, 
                             &capture1, &median, &conv1, wid1,
                             &resize };
    
    ErrorHandle errHdl(wid0, wid1);
    tdv::WorkUnitRunner runner(wus, 8, &errHdl);
    runner.run();
    
    int r = app.exec();    
    
    capture0.finish();
    capture1.finish();
    
    runner.join();
    
    return r;
}


