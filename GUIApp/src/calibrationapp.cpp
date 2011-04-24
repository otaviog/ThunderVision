#include <QApplication>
#include <QProgressDialog>
#include <tdvision/tdvcontext.hpp>
#include <tdvbasic/log.hpp>
#include <tdvision/capturestereoinputsource.hpp>
#include "cmdline.hpp"
#include "calibrationdialog.hpp"

int main(int argc, char *argv[])
{
    tdv::TdvGlobalLogDefaultOutputs();
    QApplication qapp(argc, argv);
    Q_INIT_RESOURCE(resources);
    
    tdv::StereoInputSource *inputSrc = NULL;
    tdv::ThunderSpec *spec = NULL;
    try
    {
        CMDLine cmd(argc, argv);        
        QProgressDialog loadDlg("Loading", "", 0, 0, NULL, 
                                Qt::FramelessWindowHint);
        loadDlg.show();
        try
        {
            inputSrc = cmd.createInputSource();
            spec = cmd.createSpec();
        }
        catch (const tdv::Exception &ex)
        {
            std::cout<<ex.what()<<std::endl;
            exit(1);
        }        

        if ( inputSrc == NULL )
        {
            tdv::CaptureStereoInputSource *csis 
                = new tdv::CaptureStereoInputSource;
            csis->init();                
            inputSrc = csis;
        }
    }
    catch (const tdv::Exception &unknowOpt)
    {
        std::cout<<unknowOpt.what()<<std::endl;        
        return 1;
    }
    
    tdv::TDVContext context;
    context.spec(spec);            
    context.start(inputSrc);
    
    tdv::Calibration *calib = context.runCalibration();
    
    CalibrationDialog *calibDlg = new CalibrationDialog(calib);    
    calibDlg->init();
    calibDlg->show();
    
    const int retExec = calibDlg->exec();
    
    calibDlg->dispose();
    context.dispose();
    
    return retExec;
}
