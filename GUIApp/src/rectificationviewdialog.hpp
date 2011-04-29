#ifndef TDV_RECTIFICATIONVIEW_HPP
#define TDV_RECTIFICATIONVIEW_HPP

#include <tdvision/reconstruction.hpp>
#include "ui_rectificationviewdialog.h"
#include "videowidget.hpp"

class RectificationViewDialog: public QDialog, 
                               private Ui::RectificationViewDialog
{
    Q_OBJECT;
    
public:
    RectificationViewDialog(tdv::Reconstruction *rctx, 
                            QWidget *parent = NULL);

    void init();
    
    void dispose();
                  
public slots:
    
protected:
    void closeEvent(QCloseEvent *event);        

private:    
    tdv::Reconstruction *m_rctx;
    VideoWidget *m_leftVidWid;
    VideoWidget *m_rightVidWid;
};

#endif /* TDV_RECTIFICATIONVIEW_HPP */
