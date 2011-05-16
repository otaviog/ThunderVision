/**
 * Java-like IO classes
 */
#ifndef UD_DATASTREAM_HPP
#define UD_DATASTREAM_HPP

#include <fstream>
#include <string>
#include "exception.hpp"
#include "common.hpp"

UD_NAMESPACE_BEGIN

class Matrix44f;
class Vec2f;
class Vec3f;
class Vec4f;
class Quatf;
class Planef;
class Color;
class Aabb;
class BoundingSphere;

class ScopedFileStream
{
public:
    ScopedFileStream(const std::string &fileName, std::ios_base::openmode mode);
    ~ScopedFileStream();

    std::fstream& get()
    {
        return m_stRef;
    }

private:
    std::fstream m_stRef;
};

class CommonDataStream
{
public:
    CommonDataStream(std::fstream &stream);

    bool isReadEOF();

    void writeInt(int v);
    void writeByte(Uint8 v);
    void writeFloat(float v);
    void writeString(const std::string &v);
    void writeMatrix(const Matrix44f &m);
    void writeColor(const Color &col);
    void writeVec2f(const Vec2f &v);
    void writeVec3f(const Vec3f &v);
    void writeVec4f(const Vec4f &v);
    void writeAabb(const Aabb &b);
    void writeBSphere(const BoundingSphere &s);
    void writePlane(const Planef &p);

    int readInt();
    Uint8 readByte();
    float readFloat();
    std::string readString();
    Matrix44f readMatrix();
    Color readColor();
    Vec3f readVec3f();
    Vec2f readVec2f();
    Vec4f readVec4f();
    Aabb readAabb();
    BoundingSphere readBoundingSphere();
    Planef readPlane();

private:    
    void inverseMem(Uint8 *v, size_t s);
    Uint32 inverseInt(Uint32 v);

    float inverseFloat(float v);

    std::fstream &m_st;
    size_t m_fileSize;
};

UD_NAMESPACE_END

#endif
