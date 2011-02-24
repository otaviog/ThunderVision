#ifndef TDV_MEM_HPP
#define TDV_MEM_HPP

#include <map>
#include <cuda_runtime.h>
#include <tdvbasic/common.hpp>
#include <boost/shared_ptr.hpp>
#include <cv.h>
#include "dim.hpp"

TDV_NAMESPACE_BEGIN

typedef int CudaDevId;

class FloatImageImpl
{
    typedef std::map<CudaDevId, float*> DevMemMap;

public:
    enum HostType
    {
        CPU, DEV
    };

    FloatImageImpl();

    FloatImageImpl(CvArr *mem);

    void initDev(const Dim &dim);

    void initCPU(const Dim &dim);

    float* waitDevMem();

    _IplImage* waitCPUMem();

    void memRelease();

    void dispose();

    const Dim &dim() const
    {
        return m_dim;
    }

    size_t sizeb() const
    {
        return m_dim.size()*sizeof(float);
    }

private:
    DevMemMap m_devmap;
    _IplImage *m_cpuMem;
    CudaDevId m_syncDev;
    Dim m_dim;
};

class FloatImage
{
public:
    FloatImage()
        : m_impl((FloatImageImpl*) NULL)
    {
    }

    FloatImage(CvArr *mem)
        : m_impl(new FloatImageImpl(mem))
    {
    }
    
    static FloatImage CreateCPU(const Dim &dim)
    {
        FloatImageImpl *impl = new FloatImageImpl;
        impl->initCPU(dim);

        return FloatImage(impl);
    }

    static FloatImage CreateDev(const Dim &dim)
    {
        FloatImageImpl *impl = new FloatImageImpl;
        impl->initDev(dim);

        return FloatImage(impl);
    }

    float* waitDevMem()
    {
        return m_impl->waitDevMem();
    }

    _IplImage* waitCPUMem()
    {
        return m_impl->waitCPUMem();
    }

    void memRelease()
    {
        m_impl->memRelease();
    }

    void dispose()
    {
        m_impl->dispose();
    }

    const Dim &dim() const
    {
        return m_impl->dim();
    }

    size_t sizeb() const
    {
        return m_impl->sizeb();
    }

private:
    FloatImage(FloatImageImpl *p)
        : m_impl(p)
    {
    }

    boost::shared_ptr<FloatImageImpl> m_impl;
};

TDV_NAMESPACE_END

#endif /* TDV_MEM_HPP */
