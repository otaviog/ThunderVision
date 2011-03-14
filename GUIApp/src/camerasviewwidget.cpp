#include <QHBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include "camerawidget.hpp"
#include "camerasviewwidget.hpp"
#include <highgui.h>

CamerasViewWidget::CamerasViewWidget(tdv::ReadPipe<IplImage*> *leftPipe, tdv::ReadPipe<IplImage*> *rightPipe)
{
    leftCamW = new CameraWidget(leftPipe, false);
    rightCamW = new CameraWidget(rightPipe, false);
    
    QHBoxLayout *box = new QHBoxLayout;
    box->addWidget(leftCamW);
    box->addWidget(rightCamW);
    setMinimumSize(400, 400);
    setLayout(box);
}

void CamerasViewWidget::start()
{
    tdv::Process *procs[] = { leftCamW, rightCamW, NULL };
    tdv::ProcessRunner runner(procs, this);
    
    runner.run();
}

void CamerasViewWidget::stop()
{
    
}

void CamerasViewWidget::photoChar()
{
    IplImage *limage = leftCamW->lastFrame();
    IplImage *rimage = rightCamW->lastFrame();

    QString errMessage;
    
    if ( limage == NULL && rimage == NULL )
    {
        errMessage = tr("The image of both cameras are not available");
    }
    else if ( limage == NULL )
    {
        errMessage = tr("The image of left camera is not available");
        cvReleaseImage(&rimage);
    }
    else if ( rimage == NULL )
    {
        errMessage = tr("The image of right camera is not available");
        cvReleaseImage(&limage);
    }
    
    if ( limage == NULL || rimage == NULL )
    {
        QMessageBox::warning(
            this, 
            tr("Fail to save photo"), 
            errMessage);
        
        return ;
    }
        
    QString filename = QFileDialog::getSaveFileName(
        this, tr("Save File"),
        QString(), tr("Images (*.png *.jpg)"));

    if ( !filename.isEmpty() )
    {
        QString leftFName, rightFName;
        
        int lastPeriod = filename.lastIndexOf('.');
        if ( lastPeriod >= 0 )
        {
            QString fname = filename.left(lastPeriod);
            QString ext = filename.right(filename.size() - lastPeriod);
            
            leftFName = fname + "-left" + ext;
            rightFName = fname + "-right" + ext;
        }        
        else
        {
            leftFName = filename + "-left.png";
            rightFName = filename + "-right.png";
        }
        
        cvSaveImage(leftFName.toStdString().c_str(), limage);
        cvSaveImage(rightFName.toStdString().c_str(), rimage);
        
        cvReleaseImage(&limage);
        cvReleaseImage(&rimage);
    }    
}

void CamerasViewWidget::errorOcurred(const std::exception &err)
{
    
}
