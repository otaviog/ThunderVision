#include <tdvbasic/log.hpp>
#include <QApplication>
#include <videowidget.hpp>
#include <tdvision/pipe.hpp>
#include <tdvision/captureproc.hpp>
#include <tdvision/imagesink.hpp>
#include <tdvision/processrunner.hpp>
#include <tdvision/processgroup.hpp>
#include <tdvision/workunitprocess.hpp>
#include <tdvision/exceptionreport.hpp>

class ErrorHandle: public tdv::ExceptionReport
{
public:
    ErrorHandle(VideoWidget *wid0)
        : w0(wid0)
    {
    }
    
    void errorOcurred(const std::exception &err)
    {
        std::cout<<err.what()<<std::endl;
        w0->close();
    }
    
private:
    VideoWidget *w0;
};

int main(int argc, char *argv[])
{
    tdv::TdvGlobalLogDefaultOutputs();
    
    QApplication app(argc, argv);
    
    tdv::CaptureProc capture;
    
    //capture.init("../../res/cam0.avi");
    capture.init(0);
    
    VideoWidget *wid0 = new VideoWidget;
    ErrorHandle errHdl(wid0);
    
    wid0->input(capture.output(), true);
    wid0->init();
    
    wid0->show();    
    
    tdv::ArrayProcessGroup procs;
    procs.addProcess(&capture);        
    
    tdv::ProcessRunner runner(procs, &errHdl);
    runner.run();
    
    int r = app.exec();    
    
    capture.finish();    
    wid0->dispose();
    runner.join();
    
    return r;
}
