#ifndef UD_SINGLETON_HPP
#define UD_SINGLETON_HPP

#include "common.hpp"

UD_NAMESPACE_BEGIN

template<typename T>
class DynamicSingleton
{
public:
    static T& Get()
    {
        if ( m_instance == NULL )
            m_instance = new T;

        return *m_instance;
    }

    static void Uninstancie()
    {
        delete m_instance;
        m_instance = NULL;
    }

    virtual ~DynamicSingleton() { }

protected:
    DynamicSingleton() { }
    DynamicSingleton(const DynamicSingleton &cpy) { }

private:
    DynamicSingleton& operator=(const DynamicSingleton &rhs) { }

    static T *m_instance;
};

template<typename T>
T* DynamicSingleton<T>::m_instance = NULL;

template<typename T>
class InitSingleton
{
public:
    static T& Get()
    {
        return *m_instance;
    }

    static void InitializeInstance()
    {
        m_instance = new T;
    }

    static void Uninstancie()
    {
        delete m_instance;
    }

protected:
    InitSingleton() { }
    InitSingleton(const InitSingleton &cpy) { }

private:
    InitSingleton& operator=(const InitSingleton &rhs) { }

    static T *m_instance;
};

template<typename T>
T* InitSingleton<T>::m_instance = NULL;

UD_NAMESPACE_END

#endif
