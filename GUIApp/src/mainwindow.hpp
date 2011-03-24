#ifndef TDV_MAINWINDOW_HPP
#define TDV_MAINWINDOW_HPP

#include <QMainWindow>
#include "ui_mainwindow.h"
#include "videowidget.hpp"

class MainWindow: public QMainWindow, private Ui::MainWindow
{
    Q_OBJECT;
public:
    MainWindow();
    
    virtual ~MainWindow();        
};

#endif /* TDV_MAINWINDOW_HPP */
