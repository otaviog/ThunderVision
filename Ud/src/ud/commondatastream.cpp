#include <boost/detail/endian.hpp>
#include <boost/system/system_error.hpp>
#include <string.h>
#include "math/matrix44.hpp"
#include "math/vec.hpp"
#include "ugl/color.hpp"
#include "ugl/aabb.hpp"
#include "ugl/boundingsphere.hpp"
#include "debug.hpp"
#include "commondatastream.hpp"

UD_NAMESPACE_BEGIN

//==========================
//== ScopedFileStream
//=========================
ScopedFileStream::ScopedFileStream(const std::string &filename, std::ios_base::openmode mode)
        : m_stRef(filename.c_str(), mode)
{ }

ScopedFileStream::~ScopedFileStream()
{
    m_stRef.close();
}

//==========================
//== CommonDataStream
//=========================
CommonDataStream::CommonDataStream(std::fstream &stream)
        : m_st(stream)
{
    size_t currG = m_st.tellg();

    m_st.seekg(0, std::ios::end);
    m_fileSize = m_st.tellg();

    m_st.seekg(currG, std::ios::beg);
}

bool CommonDataStream::isReadEOF()
{
    return m_fileSize == m_st.tellg();
}

void CommonDataStream::inverseMem(Uint8 *v, size_t s)
{
    Uint8 *t = new Uint8[s];
    memcpy(t, v, s);

    for (size_t i=0; i<s; i++)
        v[s-i-1] = t[i];
    
    delete [] t;
}

Uint32 CommonDataStream::inverseInt(Uint32 v)
{
    const Uint32 h1 = v & 0xff000000;
    const Uint32 h2 = v & 0x00ff0000;
    const Uint32 l1 = v & 0x0000ff00;
    const Uint32 l2 = v & 0x000000ff;

    return ((l2 << 24) | (l1 << 8)) | ((h2 >> 8) | (h1 >> 24));
}

/**
 * http://www.gamedev.net/reference/articles/article2091.asp
 */
float CommonDataStream::inverseFloat(float v)
{
    union
    {
        float f;
        unsigned char b[4];
    } d1, d2;

    d1.f = v;
    d2.b[0] = d1.b[3];
    d2.b[1] = d1.b[2];
    d2.b[2] = d1.b[1];
    d2.b[3] = d1.b[0];
    return d2.f;
}

namespace bsys = boost::system;

int CommonDataStream::readInt()
{
    int v;
    m_st.read(reinterpret_cast<char*>(&v), 4);
    if ( !m_st.good() )
        throw bsys::system_error(bsys::error_code(errno, bsys::system_category));
    
#ifdef BOOST_LITTLE_ENDIAN
    v = inverseInt(v);
#endif
    return v;
}

void CommonDataStream::writeInt(int v)
{
#ifdef BOOST_LITTLE_ENDIAN
    v = inverseInt(v);
#endif
    m_st.write(reinterpret_cast<const char*>(&v), 4);
    if ( !m_st.good() )
        throw bsys::system_error(bsys::error_code(errno, bsys::system_category));

}

float CommonDataStream::readFloat()
{
    float v;
    m_st.read(reinterpret_cast<char*>(&v), 4);
    if ( !m_st.good() )
        throw bsys::system_error(bsys::error_code(errno, bsys::system_category));

#ifdef BOOST_LITTLE_ENDIAN
    v = inverseFloat(v);
#endif

    return v;
}

void CommonDataStream::writeFloat(float v)
{
#ifdef BOOST_LITTLE_ENDIAN
    v = inverseFloat(v);
#endif
    m_st.write(reinterpret_cast<const char*>(&v), 4);
    if ( !m_st.good() )
        throw bsys::system_error(bsys::error_code(errno, bsys::system_category));    
}

Uint8 CommonDataStream::readByte()
{
    Uint8 v;
    m_st.read(reinterpret_cast<char*>(&v), 1);
    if ( !m_st.good() )
        throw bsys::system_error(bsys::error_code(errno, bsys::system_category));
        
    return v;
}

void CommonDataStream::writeByte(Uint8 v)
{
    m_st.write(reinterpret_cast<const char*>(&v), 1);
    if ( !m_st.good() )
        throw bsys::system_error(bsys::error_code(errno, bsys::system_category));

}

std::string CommonDataStream::readString()
{
    std::string v;
    int s = readInt();
    for (int i=0; i<s; i++)
    {
        const char c = readByte();
        v.push_back(c);
    }
    return v;
}

void CommonDataStream::writeString(const std::string &v)
{
    writeInt(v.size());
    for (size_t i=0; i<v.size(); i++)
        writeByte(v[i]);
}

Matrix44f CommonDataStream::readMatrix()
{
    Matrix44f r;

    for (int k=0; k<16; k++)
        r[k] = readFloat();

    return r;
}

void CommonDataStream::writeMatrix(const Matrix44f &m)
{
    for (int k=0; k<16; k++)
        writeFloat(m[k]);
}

Color CommonDataStream::readColor()
{
    Color col;
    col[0] = readFloat();
    col[1] = readFloat();
    col[2] = readFloat();
    col[3] = readFloat();
    return col;
}

void CommonDataStream::writeColor(const Color &col)
{
    writeFloat(col[0]);
    writeFloat(col[1]);
    writeFloat(col[2]);
    writeFloat(col[3]);
}

Vec2f CommonDataStream::readVec2f()
{
    Vec2f v;
    v[0] = readFloat();
    v[1] = readFloat();
    return v;
}

void CommonDataStream::writeVec2f(const Vec2f &v)
{
    writeFloat(v[0]);
    writeFloat(v[1]);
}

Vec3f CommonDataStream::readVec3f()
{
    Vec3f v;
    v[0] = readFloat();
    v[1] = readFloat();
    v[2] = readFloat();
    return v;
}

void CommonDataStream::writeVec3f(const Vec3f &v)
{
    writeFloat(v[0]);
    writeFloat(v[1]);
    writeFloat(v[2]);
}

Vec4f CommonDataStream::readVec4f()
{
    Vec4f v;
    v[0] = readFloat();
    v[1] = readFloat();
    v[2] = readFloat();
    v[3] = readFloat();
    return v;
}

void CommonDataStream::writeVec4f(const Vec4f &v)
{
    writeFloat(v[0]);
    writeFloat(v[1]);
    writeFloat(v[2]);
    writeFloat(v[3]);
}

Aabb CommonDataStream::readAabb()
{
    const Vec3f min = readVec3f();
    const Vec3f max = readVec3f();
    return Aabb(min, max);
}

void CommonDataStream::writeAabb(const Aabb &b)
{
    writeVec3f(b.getMin());
    writeVec3f(b.getMax());
}

BoundingSphere CommonDataStream::readBoundingSphere()
{
    const Vec3f center = readVec3f();
    const float radius = readFloat();
    return BoundingSphere(center, radius);
}

void CommonDataStream::writeBSphere(const BoundingSphere &s)
{
    writeVec3f(s.center());
    writeFloat(s.radius());
}

Planef CommonDataStream::readPlane()
{
    const Vec3f n = readVec3f();
    const float d = readFloat();
    return Planef(n, d);
}

void CommonDataStream::writePlane(const Planef &p)
{
    writeVec3f(p.normal);
    writeFloat(p.d);
}

UD_NAMESPACE_END
