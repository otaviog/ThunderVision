#include "cuerr.hpp"
#include "floatimage.hpp"
#include "misc.hpp"
#include <cv.h>
#include <tdvbasic/log.hpp>

TDV_NAMESPACE_BEGIN

FloatImageImpl::FloatImageImpl()
    : m_dim(-1)
{
    m_syncDev = -2;
    m_cpuMem = NULL;
}

FloatImageImpl::FloatImageImpl(CvArr *img)
    : m_dim(-1)
{
    CvSize size(cvGetSize(img));
    m_dim = Dim(size.width, size.height);
    
    m_cpuMem = misc::create32FGray(img);    
    m_syncDev = -1;
}

float* FloatImageImpl::createDevMem()
{
    CUerrExp err;
    
    float *mem;
    err << cudaMalloc((void**) &mem, sizeb());
    
    return mem;
}

void FloatImageImpl::initDev(const Dim &dim)
{
    m_dim = dim;

    float *mem = createDevMem();
    
    CudaDevId device;
    CUerrDB(cudaGetDevice(&device));
    m_devmap[device] = mem;

    m_syncDev = device;
}

void FloatImageImpl::initCPU(const Dim &dim)
{
    m_dim = dim;
    m_cpuMem = cvCreateMat(dim.height(), dim.width(), CV_32F);
}

float* FloatImageImpl::devMem()
{
    float *devMem = NULL;
    CudaDevId device;
    CUerrExp cuerr;

    CUerrDB(cudaGetDevice(&device));

    if ( m_syncDev != device )
    {
        DevMemMap::iterator mIt = m_devmap.find(device);
        if ( mIt == m_devmap.end() )
        {
            devMem = createDevMem();
            m_devmap[device] = devMem;
        }
        else
        {
            devMem = mIt->second;
        }

        if ( m_syncDev >= 0 ) // On a GPU
        {
            float *otherDevMem = m_devmap[device];
            assert(otherDevMem != NULL);
            cuerr << cudaMemcpy(devMem, otherDevMem, sizeb(),
                                cudaMemcpyDeviceToDevice);
        }
        else // On the cpu
        {
            for (int row=0; row<m_cpuMem->rows; row++)
            {
                cuerr << cudaMemcpy(devMem + row*m_cpuMem->width, 
                                    m_cpuMem->data.ptr + row*m_cpuMem->step,
                                    m_cpuMem->width*sizeof(float),
                                    cudaMemcpyHostToDevice);
            }            
        }
    }
    else
    {
        devMem = m_devmap[device];
        assert(devMem != NULL);
    }

    m_syncDev = device;
    return devMem;
}

CvMat* FloatImageImpl::cpuMem()
{
    CUerrExp cuerr;
        
    if ( m_syncDev >= 0 )
    {
        const float * const devMem = m_devmap[m_syncDev];
        assert(devMem != NULL);
        
        if ( m_cpuMem == NULL )
        {
            m_cpuMem = cvCreateMat(m_dim.height(), m_dim.width(), CV_32F);
        }
        
        for (size_t row=0; row<m_dim.height(); row++)
        {
            cudaMemcpy(m_cpuMem->data.ptr + row*m_cpuMem->step, 
                       devMem + row*m_dim.width(),
                       m_dim.width()*sizeof(float),
                       cudaMemcpyDeviceToHost);
        }

        m_syncDev = -1;      
    }

    return m_cpuMem;
}

void FloatImageImpl::dispose()
{    
    if ( m_cpuMem != NULL )
    {
        cvReleaseMat(&m_cpuMem);
        m_cpuMem = NULL;
    }

    for (DevMemMap::iterator dIt = m_devmap.begin(); dIt != m_devmap.end();
         dIt++)
    {
        cudaFree(dIt->second);
    }
    
    m_devmap.clear();
}

TDV_NAMESPACE_END
