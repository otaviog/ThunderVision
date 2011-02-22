#ifndef TDV_DIM_HPP
#define TDV_DIM_HPP

#include <tdvbasic/common.hpp>
#include <cassert>

TDV_NAMESPACE_BEGIN

class Dim
{
public:
    Dim(size_t w)
    {
        dim[0] = w;
        ndim = 1;
    }

    Dim(size_t w, size_t h)
    {
        dim[0] = w;
        dim[1] = h;
        ndim = 2;
    }

    Dim(size_t w, size_t h, size_t d)
    {
        dim[0] = w;
        dim[1] = h;
        dim[2] = d;
        ndim = 3;
    }

    size_t width() const
    {
        return dim[0];
    }

    size_t height() const
    {
        assert(ndim > 0);
        return dim[1];
    }

    size_t depth() const
    {
        assert(ndim > 1);
        return dim[2];
    }

    size_t N() const
    {
        return ndim;
    }

    size_t size() const
    {
        size_t sz = 1;
        for (size_t k=0; k<ndim; k++)
            sz *= dim[k];

        return sz;
    }

private:
    size_t dim[3];
    size_t ndim;
};

TDV_NAMESPACE_END

#endif /* TDV_DIM_HPP */
