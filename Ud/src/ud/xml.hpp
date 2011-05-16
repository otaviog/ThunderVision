#ifndef UD_XML_HPP
#define UD_XML_HPP

#include <cstdlib>
#include <boost/filesystem/path.hpp>
#include "common.hpp"
#include "tinyxml.h"

UD_NAMESPACE_BEGIN

class Color;
class Aabb;
class Quatf;
class Vec2f;
class Vec3f;
class Matrix44f;

namespace xml
{
    float nextFloat(const char *text, int &start);

    void nextFloats(const char* text, float values[],
                    int num_values, int &start);

    int nextInt(const char *text, int &start);

    void nextInts(const char* text, int values[],
                  int num_values, int &start);

    void readQuat(const TiXmlElement *elem, Quatf *quat);
    void readVec2f(const TiXmlElement *elem, Vec2f *vec);
    void readVec3f(const TiXmlElement *elem, Vec3f *vec);
    void readMatrix44f(const TiXmlElement *elem, Matrix44f *mat);
    void readColor(const TiXmlElement *elem, Color *col);
    bool readPath(const TiXmlElement *elem, std::string *path);
    void readAabb(const TiXmlElement *elem, Aabb *box);

    bool floatAttribute(const TiXmlElement *elem, const char *attrName, float &retAttr);

    const TiXmlElement* findFirstElement(const TiXmlElement *elem, const char *name);
    const TiXmlElement* findFirstElement(const TiXmlElement *elem, const char *name,
                                         const char *attr, const char *attrVal);

    void writeAabb(TiXmlElement *parent, const Aabb &box);
    void writeVec3f(TiXmlElement *parent, const std::string &name,
                    const Vec3f &v);
    void writeColor(TiXmlElement *parent, const std::string &name,
                    const Color &color);
    bool xmlBoolean(const char *attr);
}

UD_NAMESPACE_END

#endif
