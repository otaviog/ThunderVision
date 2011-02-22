#ifndef _READFILE_HPP
#define _READFILE_HPP

#include <istream>
#include <string>
#include "common.hpp"

TDV_NAMESPACE_BEGIN

namespace readfile
{
    struct Filedata
    {
        size_t size;
        char *data;

        Filedata()
        {
            data = NULL;
            size = 0;
        }

        void dispose()
        {
            delete [] data;
        }

        bool isNull() const
        {
            return data == NULL;
        }
    };

    Filedata readfile(const std::string &filename);

    Filedata readstream(std::istream &stream);
}

TDV_NAMESPACE_END

#endif /* _READFILE_HPP */
