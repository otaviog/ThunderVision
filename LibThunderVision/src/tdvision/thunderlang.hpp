#ifndef TDV_THUNDERLANG_HPP
#define TDV_THUNDERLANG_HPP

#include <tdvbasic/common.hpp>
#include <string>
#include <list>
#include <map>
#include "camerasdesc.hpp"

TDV_NAMESPACE_BEGIN

class ThunderLang
{
public:
    typedef std::map<std::string, CamerasDesc> CamerasDescMap;
    
    CamerasDesc& camerasDesc(const std::string &name)
    {
        return m_cdmap[name];
    }
    
private:
    CamerasDescMap m_cdmap;
};

class ThunderLangParser
{
public:    
    struct Error
    {
        const static int MAXFILENAME = 50;
        const static int MAXDESCRIPTION = 150;

        char filename[MAXFILENAME];
        char description[MAXDESCRIPTION];
        int linenum, column;
    };
    
    typedef std::list<Error> ErrorList;
    
    ThunderLangParser(ThunderLang &ctx)
        : m_context(ctx)
    { }
                     
    void parseFile(const std::string &filename);
    
    void ___private_ADD_ERROR(const Error &error)
    {
        m_errors.push_back(error);
    }
    
    ThunderLang& context()
    {
        return m_context;
    }
    
private:    
    ErrorList m_errors;
    ThunderLang &m_context;
};

class ThunderLangWriter
{
public:
    void write();
    
private:
};

TDV_NAMESPACE_END

#endif /* TDV_THUNDERLANG_HPP */
