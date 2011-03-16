#ifndef TDV_ERRORREPORT_HPP
#define TDV_ERRORREPORT_HPP

#include <tdvision/exceptionreport.hpp>
#include <QObject>

class ErrorReport: public QObject, public tdv::ExceptionReport
{
    Q_OBJECT;
    
public:
    void errorOcurred(const std::exception &ex)
    {
        Q_EMIT informError(QString(ex.what()));
    }
        
signals:
    void informError(QString message);
    
private:
};


#endif /* TDV_ERRORREPORT_HPP */
