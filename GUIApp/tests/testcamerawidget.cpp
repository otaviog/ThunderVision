#include <tdvbasic/log.hpp>
#include <QApplication>
#include <camerawidget.hpp>
#include <tdvision/pipe.hpp>
#include <tdvision/capturewu.hpp>
#include <tdvision/pipebuilder.hpp>
#include <tdvision/imagesink.hpp>

int main(int argc, char *argv[])
{
    tdv::TdvGlobalLogDefaultOutputs();
    
    QApplication app(argc, argv);
    
    tdv::ReadWritePipe<IplImage*, IplImage*> imgPipe0;
    tdv::ReadWritePipe<IplImage*, IplImage*> imgPipe1;
    
    CameraWidget *wid0 = new CameraWidget(&imgPipe0);
    CameraWidget *wid1 = new CameraWidget(&imgPipe1);

    wid0->init();
    wid1->init();
    
    tdv::PipeBuilder pipeline, pipe2;
    
    tdv::CaptureWU *capture0 = new tdv::CaptureWU(0);    
    tdv::CaptureWU *capture1 = new tdv::CaptureWU(1);    
    
    capture0->colorImage(&imgPipe0);
    pipeline >> capture0 >> new tdv::ImageSink;
    pipeline.run();
    
    capture1->colorImage(&imgPipe1);
    pipe2 >> capture1 >> new tdv::ImageSink;
    pipe2.run();

    wid0->show();
    wid1->show();
    
    int r = app.exec();

    imgPipe0.finish();
    imgPipe1.finish();
    
    capture0->endCapture();
    capture1->endCapture();

    pipeline.join();
    pipe2.join();
    return r;
}


