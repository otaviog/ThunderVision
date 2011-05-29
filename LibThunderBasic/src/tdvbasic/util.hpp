#ifndef TDV_UTIL_HPP
#define TDV_UTIL_HPP

#include <cstdlib>
#include "common.hpp"

TDV_NAMESPACE_BEGIN

namespace util
{
    /**
     * Replaces the first occurrence of a string by another.
     * @param orign the original string.
     * @param oSz the length of the original string (not counting the
     * terminating \0).
     * @param search the search string to be replaced.
     * @param sSz the length of the search string (not clouting the
     * terminating the \0).
     * @param replace the replacement string.
     * @param rSz the length of the replacement string (not cutting
     * the terminating \0).
     * @return a new string with the the first occurrence replaced. The
     * string is allocated with new [] operator and must deallocated
     * with delete [] operator. If no search string is found then NULL
     * is returned.
     */
    char *strReplaceFirst(const char *orign, size_t oSz,
                          const char *search, size_t sSz,
                          const char *replace, size_t rSz);

    /**
     * Replaces the occurrences of a string by another.
     * @param orign the original string.
     * @param search the search string to be replaced.
     * @param replace the replacement string.
     * @param rSz the length of the replacement string (not cutting
     * the terminating \0).
     * @return a new string with the the occurrences replaced. The
     * string is allocated with new [] operator and must deallocated
     * with delete [] operator.If no search string is found then NULL
     * is returned.
     */
    char *strReplace(const char *orign, const char *search,
                     const char *replace, size_t rLen);

    /**
     * Computes the next power of 2 a number, 
     * according to bithacks 
     * (http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2).
     * @param v the number to be round up.
     */
    size_t nextPowerOf2(size_t v);
    
    size_t previousPowerOf2(size_t v);

    size_t nearestPowerOf2(size_t v);    
    
    void logBacktrace();
}

TDV_NAMESPACE_END

#endif /* TDV_UTIL_HPP */
