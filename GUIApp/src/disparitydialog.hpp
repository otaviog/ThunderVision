#ifndef TDV_DISPARITYDIALOG_HPP
#define TDV_DISPARITYDIALOG_HPP

#include <QDialog>
#include <tdvision/reconstruction.hpp>
#include "ui_disparitydialog.h"

class VideoWidget;

class DisparityDialog: public QDialog,
                       private Ui::DisparityDialog
{
    Q_OBJECT;
    
public:
    DisparityDialog(tdv::Reconstruction *rctx, QWidget *parent = NULL);
    
    void init();
    
    void dispose();    
    
protected:
    void closeEvent(QCloseEvent *event);
    
private:
    VideoWidget *m_dispView;
    tdv::Reconstruction *m_recContext;    
};

#endif /* TDV_DISPARITYDIALOG_HPP */
