#ifndef UD_COMMON_HPP
#define UD_COMMON_HPP

#include <algorithm>

#define UD_NAMESPACE ud
#define UD_NAMESPACE_BEGIN namespace UD_NAMESPACE {
#define UD_NAMESPACE_END }

#ifdef UD_DLL_COMPILATION
#define UD_EXPORT __declspec(dllexport)
#else
#define UD_EXPORT
#endif

#if defined(__INTEL_COMPILER)
#pragma warning(disable:981)
#pragma warning(disable:869)
#endif

#if !defined(WIN32) || defined(__MINGW32__) || defined(__CYGWIN__)
#include <inttypes.h>
#endif

UD_NAMESPACE_BEGIN

#if !defined(WIN32) || defined(__MINGW32__) ||  defined(__CYGWIN__)
typedef int8_t Int8;
typedef uint8_t Uint8;

typedef int16_t Int16;
typedef uint16_t Uint16;

typedef int32_t Int32;
typedef uint32_t Uint32;
typedef Int32 Integer;

typedef int64_t Int64;
typedef uint64_t Uint64;
#else
typedef char Int8;
typedef unsigned char Uint8;

typedef short Int16;
typedef unsigned short Uint16;

typedef int Int32;
typedef unsigned int Uint32;

typedef long long Int64;
typedef unsigned long long Uint64;
#endif

#if defined(_MSC_VER)
template <typename T>
inline T hgMax(T a, T b)
{
    if ( a > b )
      return a;
	return b;	
}

template <typename T>
inline T hgMin(T a, T b)
{
	if ( a < b )
		return a;
	return b;
}
#else
template<typename T>
inline T hgMax(T a, T b)
{
    return std::max(a, b);
}

template <typename T>
inline T hgMin(T a, T b)
{
    return std::min(a, b);
}
#endif

#ifndef M_PI
#define M_PI        3.14159265358979323846
#endif

#define UD_FPI 3.14159265358979323846f
#define UD_PI 3.14159265358979323846

UD_NAMESPACE_END

#ifdef WIN32
#define NOMINMAX 1
#include <windows.h>
#endif

#include <GL/glew.h>
#include <GL/gl.h>

#endif
