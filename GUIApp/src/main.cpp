#include <tdvbasic/log.hpp>
#include <QApplication>
#include <tdvision/tdvcontext.hpp>
#include <tdvision/capturestereoinputsource.hpp>
#include "mainwindow.hpp"

int main(int argc, char *argv[])
{
    QApplication qapp(argc, argv);    
    tdv::TdvGlobalLogDefaultOutputs();

    tdv::CaptureStereoInputSource inputSrc;
    //inputSrc.init("../../res/cam0.avi", "../../res/cam1.avi");
    inputSrc.init();
    
    tdv::TDVContext context;
    context.start(&inputSrc);
    //context.loadSpecFrom("~/.thundervision", inputSrc);

    MainWindow *mainWindow = new MainWindow(&context);    
    mainWindow->show();        
    
    int r = qapp.exec();
    
    context.dispose();
    
    return r;
}
