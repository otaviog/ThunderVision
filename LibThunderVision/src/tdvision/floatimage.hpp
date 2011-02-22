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

class FloatImageMemImpl
{
    typedef std::map<CudaDevId, float*> DevMemMap;

public:
    enum HostType
    {
        CPU, DEV
    };

    FloatImageMemImpl();

    FloatImageMemImpl(CvArr *mem);

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

class FloatImageMem
{
public:
    FloatImageMem()
        : m_impl((FloatImageMemImpl*) NULL)
    {
    }

    FloatImageMem(CvArr *mem)
        : m_impl(new FloatImageMemImpl(mem))
    {
    }
    
    static FloatImageMem CreateCPU(const Dim &dim)
    {
        FloatImageMemImpl *impl = new FloatImageMemImpl;
        impl->initCPU(dim);

        return FloatImageMem(impl);
    }

    static FloatImageMem CreateDev(const Dim &dim)
    {
        FloatImageMemImpl *impl = new FloatImageMemImpl;
        impl->initDev(dim);

        return FloatImageMem(impl);
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
    FloatImageMem(FloatImageMemImpl *p)
        : m_impl(p)
    {
    }

    boost::shared_ptr<FloatImageMemImpl> m_impl;
};

TDV_NAMESPACE_END

#endif /* TDV_MEM_HPP */
