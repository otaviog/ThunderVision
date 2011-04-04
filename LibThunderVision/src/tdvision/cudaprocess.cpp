#include <cuda_runtime.h>
#include <boost/scoped_array.hpp>
#include "cuerr.hpp"
#include "workunit.hpp"
#include "cudaprocess.hpp"

TDV_NAMESPACE_BEGIN

void CUDAProcess::process()
{   
    CUerrExp err;    
    err << cudaSetDevice(m_deviceId);
    
    boost::scoped_array<bool> endArray(new bool[m_units.size()]);
    for (size_t i=0; i<m_units.size(); i++)
        endArray[i] = true;
    
    bool cont = true;
    while ( cont )
    {       
        cont = false;
        for (size_t i=0; i<m_units.size(); i++)
        {
            if ( endArray[i] )
                endArray[i] = m_units[i]->update();
            
            cont = cont || endArray[i];
        }
    }
}

void CUDAProcess::finish()
{
    m_end = true;
}

TDV_NAMESPACE_END
