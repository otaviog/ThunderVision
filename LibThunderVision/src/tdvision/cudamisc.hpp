#ifndef TDV_CUDAMISC_HPP
#define TDV_CUDAMISC_HPP

#include "dim.hpp"

inline dim3 tdvDimTo(const tdv::Dim &tdim)
{
    return dim3(tdim.width(), tdim.height(), tdim.depth());
}

#endif /* TDV_CUDAMISC_HPP */
