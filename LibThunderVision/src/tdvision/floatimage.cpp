#include "cuerr.hpp"
#include "mem.hpp"
#include <cv.h>

TDV_NAMESPACE_BEGIN

FloatImageMemImpl::FloatImageMemImpl()
    : m_dim(-1)
{
    m_syncDev = -2;
    m_cpuMem = NULL;
}

FloatImageMemImpl::FloatImageMemImpl(CvArr *img)
    : m_dim(-1)
{
    CvSize size(cvGetSize(img));
    m_dim = Dim(size.width, size.height);
    
    IplImage *grayTmp = cvCreateImage(size,
                                      IPL_DEPTH_8U, 1);
    cvCvtColor(img, grayTmp, CV_RGB2GRAY);
    m_cpuMem = cvCreateImage(size,
                             IPL_DEPTH_32F, 1);
    cvConvertScale(grayTmp, m_cpuMem, 1.0/255.0);

    cvReleaseImage(&grayTmp);

    m_syncDev = -1;
}

void FloatImageMemImpl::initDev(const Dim &dim)
{
    m_dim = dim;

    float *mem;
    CUerrDB(cudaMalloc((void**) &mem, sizeb()));

    CudaDevId device;
    CUerrDB(cudaGetDevice(&device));
    m_devmap[device] = mem;

    m_syncDev = device;
}

void FloatImageMemImpl::initCPU(const Dim &dim)
{
    m_dim = dim;
    m_cpuMem = cvCreateImage(cvSize(dim.width(), dim.height()),
                             IPL_DEPTH_32F, 1);

}

float* FloatImageMemImpl::waitDevMem()
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
            cuerr << cudaMalloc((void**) &devMem, sizeb());
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
            for (size_t row=0; row<m_cpuMem->height; row++)
            {
                cuerr << cudaMemcpy(devMem + row*m_cpuMem->width, 
                                    m_cpuMem->imageData + row*m_cpuMem->widthStep,
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

IplImage* FloatImageMemImpl::waitCPUMem()
{
    CUerrExp cuerr;
        
    if ( m_syncDev >= 0 )
    {
        const float * const devMem = m_devmap[m_syncDev];
        assert(devMem != NULL);
        
        if ( m_cpuMem == NULL )
        {
            m_cpuMem = cvCreateImage(cvSize(m_dim.width(), m_dim.height()),
                                     IPL_DEPTH_32F, 1);
        }
        
        for (size_t row=0; row<m_cpuMem->height; row++)
        {
            cudaMemcpy(m_cpuMem->imageData + row*m_cpuMem->widthStep, 
                       devMem + row*m_cpuMem->width,
                       m_cpuMem->width*sizeof(float),
                       cudaMemcpyDeviceToHost);
        }

        m_syncDev = -1;
    }

    return m_cpuMem;
}

void FloatImageMemImpl::memRelease()
{

}

void FloatImageMemImpl::dispose()
{
    if ( m_cpuMem != NULL )
    {
        cvReleaseImage(&m_cpuMem);
    }

    for (DevMemMap::iterator dIt = m_devmap.begin(); dIt != m_devmap.end();
         dIt++)
    {
        cudaFree(dIt->second);
    }

}

TDV_NAMESPACE_END
