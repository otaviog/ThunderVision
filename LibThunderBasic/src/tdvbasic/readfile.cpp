#include <fstream>
#include "readfile.hpp"

TDV_NAMESPACE_BEGIN

namespace readfile
{
    Filedata readfile(const std::string &filename)
    {
        std::ifstream infile(filename.c_str());
        Filedata contents;

        if ( infile.good() )
             contents = readstream(infile);

        infile.close();

        return contents;
    }

    Filedata readstream(std::istream &stream)
    {
        stream.seekg(0, std::ios::end);

        const int length = stream.tellg();
        Filedata contents;

        if ( length > 0 )
        {
            stream.seekg(0, std::ios::beg);
            contents.data = new char[length + 1];

            stream.read(contents.data, length);
            contents.data[length] = 0;
            contents.size = length;
        }

        return contents;
    }
}

TDV_NAMESPACE_END
