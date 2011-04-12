#include <QApplication>
#include <tdvision/tdvcontext.hpp>
#include "calibrationdialog.hpp"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);    
    
    tdv::
        
    tdv::TDVContext context;
    
    CalibrationDialog *calibDlg = new CalibrationDialog;    
    calibDlg->init();
    calibDlg->show();
    
    const int retExec = calibDlg->exec();
    calibDlg->dispose();
    
    return retExec;
}
