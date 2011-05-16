#include "ugl/aabb.hpp"
#include "ugl/color.hpp"
#include "math/math.hpp"
#include "logger.hpp"
#include "xml.hpp"

using std::string;

UD_NAMESPACE_BEGIN

namespace xml
{
    float nextFloat(const char *text, int &start)
    {
        static const int floatBufferSize = 25;

        char ch, buff[floatBufferSize + 1];
        int i = start,
            ch_i = 0;
        bool finded = false;
        float ret = 0.0;

        ch = text[i];
        while ( ch != 0 && !finded )
        {
            if ( isdigit(ch) || ch == '.' || ch == '-' || ch == 'e')
            {
                buff[ch_i%floatBufferSize] = ch;
                ch_i++;
            }
            else if ( ch_i > 0 )
                finded = true;

            ++i;
            ch = text[i];
        }

        if ( ch == 0 && ch_i > 0 )
            finded = true;

        start = i;
        if ( finded )
        {
            buff[ch_i % (floatBufferSize+1)] = 0;
            ret = static_cast<float>(atof(buff));
        }

        return ret;
    }

    void nextFloats(const char* text, float values[],
                    int num_values, int &start)
    {
        for (int i=0; i<num_values && text[start] != 0; i++)
            values[i] = nextFloat(text, start);
    }

    int nextInt(const char *text, int &start)
    {
        static const int uintBufferSize = 12;

        char ch, buff[uintBufferSize + 1];
        int ch_i = 0, i = start;
        bool finded = false;
        Uint32 ret = 0;

        ch = text[i];
        while ( ch != 0 && !finded )
        {
            if ( isdigit(ch) )
            {
                buff[ch_i%uintBufferSize] = ch;
                ch_i++;
            }
            else if ( ch_i > 0 )
                finded = true;
            ++i;
            ch = text[i];
        }

        if ( ch == 0 && ch_i > 0 )
            finded = true;

        start = i;
        if ( finded )
        {
            buff[ch_i  % (uintBufferSize+1)] = 0;
            ret = static_cast<Uint32>(atoi(buff));
        }

        return ret;
    }

    void nextInts(const char* text, int values[],
                  int num_values, int &start)
    {
        for (int i=0; i<num_values && text[start] != 0; i++)
            values[i] = nextInt(text, start);
    }

    void readQuat(const TiXmlElement *e, Quatf *q)
    {
        floatAttribute(e, "w", q->w);
        floatAttribute(e, "x", q->x);
        floatAttribute(e, "y", q->y);
        floatAttribute(e, "z", q->z);
    }

    void readVec3f(const TiXmlElement *e, Vec3f *v)
    {
        floatAttribute(e, "x", (*v)[0]);
        floatAttribute(e, "y", (*v)[1]);
        floatAttribute(e, "z", (*v)[2]);
    }

    void readVec2f(const TiXmlElement *e, Vec2f *v)
    {
        floatAttribute(e, "x", (*v)[0]);
        floatAttribute(e, "y", (*v)[1]);
    }

    void readMatrix44f(const TiXmlElement *e, Matrix44f *m)
    {
        const char *text = e->GetText();

        if ( text )
        {
            int start = 0;
            nextFloats(text, m->m, 16, start);
        }
    }

    void readColor(const TiXmlElement *e, Color *c)
    {
        float r, g, b, a;

        if ( floatAttribute(e, "r", r) )
            c->R(r);

        if ( floatAttribute(e, "g", g) )
            c->G(g);

        if ( floatAttribute(e, "b", b) )
            c->B(b);

        if ( floatAttribute(e, "a", a) )
            c->A(a);
    }

    bool readPath(const TiXmlElement *e, string *path)
    {
        if ( e == NULL )
            return false;

        const char *pathAttr = e->Attribute("path");
        bool r = false;

        if ( pathAttr )
        {
            *path = string(pathAttr);
            r = true;
        }
        return r;
    }

    void readAabb(const TiXmlElement *elem, Aabb *box)
    {
        const TiXmlElement *aux;
        Vec3f v(0.0f);

        aux = elem->FirstChildElement("min");
        if ( aux )
        {
            readVec3f(aux, &v);
            box->add(v);
        }

        aux = elem->FirstChildElement("max");
        if ( aux )
        {
            readVec3f(aux, &v);
            box->add(v);
        }
    }

    const TiXmlElement* findFirstElement(const TiXmlElement *elem, const char *name)
    {
        const TiXmlElement *ret = NULL;

        if ( elem )
        {
            if ( strcmp(elem->Value(), name) == 0 )
            {
                ret = elem;
            }
            else
            {
                while ( elem && ret == NULL)
                {
                    ret = findFirstElement(elem->FirstChildElement(), name);

                    if ( ret == NULL )
                        elem = elem->NextSiblingElement(name);
                }
            }
        }

        return ret;
    }

    const TiXmlElement* findFirstElement(const TiXmlElement *elem, const char *name,
                                         const char *attr, const char *attrVal)
    {
        const TiXmlElement *ret = NULL;
        if ( elem )
        {
            if ( strcmp(elem->Value(), name) == 0
                 && strcmp(elem->Attribute(attr), attrVal) == 0 )
            {
                ret = elem;
            }
            else
            {
                while ( elem  && ret == NULL )
                {
                    ret = findFirstElement(elem->FirstChildElement(), name);

                    if ( ret == NULL )
                        elem = elem->NextSiblingElement(name);
                }
            }
        }

        return ret;
    }

    bool floatAttribute(const TiXmlElement *elem, const char *attrName, float &retAttr)
    {
        bool ret = false;

        const char *attr = elem->Attribute(attrName);

        if ( attr )
        {
            retAttr = (static_cast<float>(atof(attr)));
            ret = true;
        }

        return ret;
    }

    void writeAabb(TiXmlElement *parent, const Aabb &box)
    {
        TiXmlElement *boxElem = new TiXmlElement("aabbox");

        writeVec3f(boxElem, "min", box.getMin());
        writeVec3f(boxElem, "max", box.getMax());

        parent->LinkEndChild(boxElem);
    }

    void writeVec3f(TiXmlElement *parent, const string &name,
                    const Vec3f &v)
    {
        TiXmlElement *vElem = new TiXmlElement(name.c_str());
        vElem->SetDoubleAttribute("x", v[0]);
        vElem->SetDoubleAttribute("y", v[1]);
        vElem->SetDoubleAttribute("z", v[2]);
        parent->LinkEndChild(vElem);
    }

    void writeColor(TiXmlElement *parent, const string &name,
                    const Color &color)
    {
        TiXmlElement *colE = new TiXmlElement(name.c_str());
        colE->SetDoubleAttribute("r", color.R());
        colE->SetDoubleAttribute("g", color.G());
        colE->SetDoubleAttribute("b", color.B());
        colE->SetDoubleAttribute("a", color.A());
        parent->LinkEndChild(colE);
    }

    bool xmlBoolean(const char *attr)
    {
        if ( strcmp(attr, "true") == 0 )
            return true;
        else
            return false;
    }
}

UD_NAMESPACE_END
