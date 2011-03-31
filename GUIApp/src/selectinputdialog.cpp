#include <QFileDialog>
#include <tdvision/capturestereoinputsource.hpp>
#include "selectinputdialog.hpp"

SelectInputDialog::SelectInputDialog()
{
    setupUi(this);
    
    connect(pbCameras, SIGNAL(clicked()),
            this, SLOT(useCameras()));
    connect(pbVideos, SIGNAL(clicked()),
            this, SLOT(useVideos()));
    connect(pbImages, SIGNAL(clicked()),
            this, SLOT(useImages()));

    m_mode = None;
}

void SelectInputDialog::useCameras()
{
    m_mode = Camera;
    accept();
}
    
void SelectInputDialog::useVideos()
{ 
    if ( openDialogs() )
    {
        m_mode = Video;
        accept();
    }
    else
        m_mode = None;
}
    
void SelectInputDialog::useImages()
{
    m_mode = Image;
}

bool SelectInputDialog::openDialogs()
{
   QString lfFileName = QFileDialog::getOpenFileName(this, tr("Open left video file"));    
    if ( lfFileName.isEmpty() )
        return false;
    
    QString rgFileName = QFileDialog::getOpenFileName(this, tr("Open right video file"));
    if ( rgFileName.isEmpty() )
        return false;
    
    m_lfFilename = lfFileName;
    m_rgFilename = rgFileName;
    
    return true;
}

tdv::StereoInputSource* SelectInputDialog::createInputSource()
{
    tdv::StereoInputSource *inputSrc = NULL;
    
    if ( m_mode == Video )
    {
        tdv::CaptureStereoInputSource *csis = new tdv::CaptureStereoInputSource;
        csis->init(m_lfFilename.toStdString(),
                   m_rgFilename.toStdString());
        inputSrc = csis;
    }
    else if ( m_mode == Image )
    {
    }
    else if ( m_mode == Camera )
    {
        tdv::CaptureStereoInputSource *csis = new tdv::CaptureStereoInputSource;
        csis->init();        
        inputSrc = csis;
    }
    
    return inputSrc;
}
