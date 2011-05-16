#ifndef UD_NONCOPYABLE_HPP
#define UD_NONCOPYABLE_HPP

#include "common.hpp"

UD_NAMESPACE_BEGIN

class NonCopyable
{
protected:
    NonCopyable() { }
    virtual ~NonCopyable() { }

private:
    NonCopyable(const NonCopyable &) { }
    NonCopyable& operator=(const NonCopyable &)
    {
        return *this;
    }
};

class NonAssignable
{
protected:
    virtual ~NonAssignable() { }

private:
    NonAssignable& operator=(const NonCopyable &)
    {
        return *this;
    }
};

UD_NAMESPACE_END

#endif
