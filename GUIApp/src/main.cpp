#include <QApplication>
#include <tdvision/thunderlang.hpp>
#include <tdvision/parserexception.hpp>
#include "appcontext.hpp"
#include "mainwindow.hpp"

int main(int argc, char *argv[])
{
    QApplication qapp(argc, argv);
    MainWindow *mainWindow = new MainWindow;
    
    AppContext appCtx;    
    appCtx.specFilename("~/.thundervision");
    appCtx.init();
    
    
    mainWindow->show();        
    
    return qapp.exec();
}
