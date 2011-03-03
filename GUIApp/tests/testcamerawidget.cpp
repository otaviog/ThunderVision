#include <tdvbasic/log.hpp>
#include <QApplication>
#include <camerawidget.hpp>
#include <tdvision/pipe.hpp>
#include <tdvision/capturewu.hpp>
#include <tdvision/imagesink.hpp>
#include <tdvision/workunitrunner.hpp>

class ErrorHandle: public tdv::WorkExceptionReport
{
public:
    void errorOcurred(const std::exception &err)
    {
        std::cout<<err.what()<<std::endl;
    }
};

int main(int argc, char *argv[])
{
    tdv::TdvGlobalLogDefaultOutputs();
    
    QApplication app(argc, argv);
    
    tdv::CaptureWU capture0(0);
    tdv::CaptureWU capture1(1);
    tdv::ImageSink sink0, sink1;
    
    CameraWidget *wid0 = new CameraWidget(capture0.colorImage());
    CameraWidget *wid1 = new CameraWidget(capture1.colorImage());            

    sink0.input(capture0.output());
    sink1.input(capture1.output());
    
    wid0->show();
    wid1->show();

    tdv::WorkUnit *wus[] = {&capture0, wid0, &capture1, wid1, 
                            &sink0, &sink1};
    
    ErrorHandle errHdl;
    tdv::WorkUnitRunner runner(wus, 6, &errHdl);
    runner.run();
    
    int r = app.exec();    
    
    capture0.finish();
    capture1.finish();
    
    runner.join();
    
    return r;
}


