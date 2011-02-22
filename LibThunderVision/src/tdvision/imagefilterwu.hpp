#ifndef TDV_IMAGEFILTERWU_HPP
#define TDV_IMAGEFILTERWU_HPP

#include <tdvbasic/common.hpp>

TDV_NAMESPACE_BEGIN

class ImageFilterWU
{
public:
    virtual void input(FloatImageMem input) = 0;

    virtual FloatImageMem output() = 0;

    virtual void compute() = 0;
private:

};

TDV_NAMESPACE_END

#endif /* TDV_IMAGEFILTERWU_HPP */
