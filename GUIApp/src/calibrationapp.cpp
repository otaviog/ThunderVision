#include <QApplication>
#include "calibrationdialog.hpp"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    CalibrationDialog *calibDlg = new CalibrationDialog;
    calibDlg->init();
    return calibDlg->exec();    
}
