#ifndef TDV_SELECTINPUTDIALOG_HPP
#define TDV_SELECTINPUTDIALOG_HPP

#include <tdvision/stereoinputsource.hpp>
#include "ui_selectinputdialog.h"

class SelectInputDialog: public QDialog, private Ui::SelectInput
{
    Q_OBJECT;    
public:
    SelectInputDialog();
    
    tdv::StereoInputSource* createInputSource();
                                          
private slots:
    void useCameras();
    
    void useVideos();
    
    void useImages();
    
private:
    bool openDialogs();
    
    enum 
    {
        Video, 
        Image,
        Camera,
        None
    } m_mode;
        
    QString m_lfFilename, m_rgFilename;
};

#endif /* TDV_SELECTINPUTDIALOG_HPP */
