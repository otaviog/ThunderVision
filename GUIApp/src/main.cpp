#include <QApplication>
#include <tdvision/thunderlang.hpp>
#include <tdvision/parserexception.hpp>
#include "mainwindow.hpp"

static void initFromSpec()
{
    try
    {
        tdv::ThunderSpec spec;
        tdv::ThunderLangParser parser(spec);
        parser.parseFile("");
    }
    catch (const tdv::ParserException &ex)
    {
    }
}

int main(int argc, char *argv[])
{
    QApplication qapp(argc, argv);
    MainWindow *mainWindow = new MainWindow;
    
    mainWindow->show();        
    
    return qapp.exec();
}
