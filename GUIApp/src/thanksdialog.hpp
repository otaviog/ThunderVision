#ifndef TDV_THANKDSDIALOG_HPP
#define TDV_THANKDSDIALOG_HPP

#include <QDialog>
#include <QThread>
#include "ui_thanks.h"

class SoundThread: public QThread
{
public:
    void run();
};

class ThanksDialog: public QDialog, public Ui::Thanks
{
    Q_OBJECT;
    
public:
    ThanksDialog(QWidget *parent = NULL);
    
public slots:
    void playThunder();
    
protected:
    virtual void showEvent(QShowEvent * event);
    
private:
    SoundThread soundThread;
};
#endif /* TDV_THANKDSDIALOG_HPP */
