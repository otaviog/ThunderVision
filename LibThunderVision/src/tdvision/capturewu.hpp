#ifndef TDV_CAPTUREWU_HPP
#define TDV_CAPTUREWU_HPP

#include <tdvbasic/common.hpp>
#include "floatimage.hpp"
#include "workunit.hpp"
#include "pipe.hpp"

TDV_NAMESPACE_BEGIN

class CaptureWU: public WorkUnit
{
public:
    CaptureWU(int device);
    
    void output(WritePipe<FloatImage> *wpipe)
    {
        m_wpipe = wpipe;
    }

    void process();
     
    void endCapture();
    
private:
    WritePipe<FloatImage> *m_wpipe;
    bool m_endCapture;
    int m_capDevice;
};

TDV_NAMESPACE_END

#endif /* TDV_CAPTUREWU_HPP */
