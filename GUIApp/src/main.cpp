#include <QApplication>
#include <QProgressDialog>
#include <boost/lexical_cast.hpp>
#include <QLocale>
#include <tdvbasic/log.hpp>
#include <tdvision/glcommon.hpp>
#include <tdvision/tdvcontext.hpp>
#include <tdvision/capturestereoinputsource.hpp>
#include "cmdline.hpp"
#include "mainwindow.hpp"

#include <iostream>

static void printCUDAInfo()
{
    using namespace std;
    
    cudaDeviceProp prop;    
    cudaGetDeviceProperties(&prop, 0);
    
    cout << "Name: " << prop.name << endl
         << "Global Mem: " << prop.totalGlobalMem/1024 << "Kib" << endl
         << "Shared Mem: " << prop.sharedMemPerBlock/1024 << "Kib" << endl
         << "Regs Block: " << prop.regsPerBlock << endl
         << "Max Threads Block:" << prop.maxThreadsPerBlock << endl
         << "Const Mem: " << prop.totalConstMem << endl
         << "Multi Proc: " << prop.multiProcessorCount << endl;        
}

int main(int argc, char *argv[])
{    
    tdv::TdvGlobalLogDefaultOutputs();    
    
    printCUDAInfo();
    
    QApplication qapp(argc, argv);        
    Q_INIT_RESOURCE(resources);

    tdv::StereoInputSource *inputSrc = NULL;
    tdv::ThunderSpec *spec = NULL;
    try
    {
        CMDLine cmd(argc, argv);
        
        QProgressDialog loadDlg("Loading", "", 0, 0, NULL, 
                                Qt::FramelessWindowHint);
        loadDlg.show();
        try
        {
            inputSrc = cmd.createInputSource();
            spec = cmd.createSpec();
        }
        catch (const tdv::Exception &ex)
        {
            std::cout<<ex.what()<<std::endl;
            loadDlg.close();
            exit(1);
        }

        loadDlg.close();
    }
    catch (const tdv::Exception &unknowOpt)
    {
        std::cout<<unknowOpt.what()<<std::endl;
        
    }        
    
    tdv::TDVContext context;
    context.spec(spec);
    
    MainWindow *mainWindow = new MainWindow(&context);    
    glewInit();
    mainWindow->start(inputSrc);
    mainWindow->show();        
    
    int r = qapp.exec();
    
    context.dispose();
    delete spec;
    
    return r;
}
