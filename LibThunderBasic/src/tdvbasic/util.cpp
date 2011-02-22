#include <cstring>
#include <vector>
#include "util.hpp"

TDV_NAMESPACE_BEGIN

namespace util
{
    char *strReplaceFirst(const char *orign, size_t oLen,
                          const char *search, size_t sLen,
                          const char *replace, size_t rLen)
    {

        const char *found = strstr(orign, search);
        char *newStr = NULL;

        if ( found != NULL )
        {
            const size_t foundIdx = found - orign;
            const size_t nLen = oLen + (rLen - sLen);
            newStr = new char[nLen + 1];
            newStr[nLen] = '\0';

            memcpy(newStr,
                   orign,
                   foundIdx);

            memcpy(newStr + foundIdx,
                   replace,
                   rLen);

            memcpy(newStr + foundIdx + rLen,
                   orign + foundIdx + sLen,
                   oLen - (foundIdx + sLen));
        }

        return newStr;
    }

    char *strReplace(const char *orign, const char *search,
                     const char *replace, size_t rLen)
    {
        using namespace std;

        const size_t oLen = strlen(orign);
        const size_t sLen = strlen(search);

        char *newStr = NULL;
        vector<size_t> foundLocs;
        size_t foundNum = 0;

        {
            const char *found = NULL;
            size_t searchStart = 0;
            while ( (found = strstr(orign + searchStart, search)) != NULL )
            {
                const size_t foundIdx = found - orign;
                foundLocs.push_back(foundIdx);
                searchStart = foundIdx + sLen;
                foundNum++;
            }
        }

        if ( !foundLocs.empty() )
        {
            const size_t nLen = oLen + (rLen - sLen)*foundNum;

            newStr = new char[nLen + 1];
            newStr[nLen] = '\0';

            size_t lastFoundEndIdx = 0,
                newStrInsertLastEndIdx = 0,
                foundCount = 0;

            memcpy(newStr, orign, foundLocs[0]);

            for (size_t found=0; found<foundLocs.size(); found++)
            {
                const size_t foundPos = foundLocs[found];
                const size_t newFoundPos = foundPos + found*(rLen - sLen);

                memcpy(newStr + newFoundPos,
                       replace,
                       rLen);

                const size_t nextFoundPos = found < (foundLocs.size() - 1)
                    ? foundLocs[found + 1]
                    : oLen;

                const size_t copyLen = nextFoundPos - (foundPos + sLen);
                memcpy(newStr + newFoundPos + rLen,
                       orign + foundPos + sLen,
                       copyLen);
            }
        }


        return newStr;
    }
}

TDV_NAMESPACE_END
