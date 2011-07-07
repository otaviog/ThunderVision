#include <QSound>
#include <QThread>
#include "thanksdialog.hpp"

void SoundThread::run()
{
    QSound::play("../struck.wav");
};

ThanksDialog::ThanksDialog(QWidget *parent)
 : QDialog(parent)
{
    setupUi(this);
    connect(pbThunder, SIGNAL(clicked()),
            this, SLOT(playThunder()));
}

void ThanksDialog::playThunder()
{
    if ( !soundThread.isRunning() )
        soundThread.start();    
}

void ThanksDialog::showEvent(QShowEvent * event)
{
    QDialog::showEvent(event);    
    playThunder();
}

