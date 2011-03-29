#include <QApplication>
#include <tdvision/tdvcontext.hpp>
#include "mainwindow.hpp"

int main(int argc, char *argv[])
{
    QApplication qapp(argc, argv);    
    
    CameraStereoInputSource inputSrc;
    TDVContext context;
    appCtx.init("~/.thundervision", inputSrc);

    MainWindow *mainWindow = new MainWindow(appCtx);    
    mainWindow->show();        
    
    return qapp.exec();
}
