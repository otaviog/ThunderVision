#include <QApplication>
#include "calibrationwidget.hpp"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    CalibrationWidget *wid = new CalibrationWidget;
    wid->show();
    
    return app.exec();
}
