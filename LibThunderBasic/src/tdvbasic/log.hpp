#ifndef TDV_LOG_HPP
#define TDV_LOG_HPP

#include <cstdio>
#include <iostream>
#include <string>
#include <map>
#include <list>

#include <boost/format.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>

#include "common.hpp"

TDV_NAMESPACE_BEGIN

class LogOutput
{
public:
    virtual void emit(const char *message, size_t size) = 0;

private:
};

class LogCategory
{
public:
    typedef std::list<boost::shared_ptr<LogOutput> > LogOutputPtrListType;

    static const size_t MAX_MESSAGE_SIZE;

    void printf(const char *format, ...)
        __attribute__ ((__format__ (__printf__, 2, 3)));

    LogCategory& operator=(const boost::format &format)
    {
        emitMessage(format.str().c_str(), format.size());
        return *this;
    }

    void addOutput(LogOutputPtrListType::value_type output)
    {
        m_logoutputList.push_back(output);
    }

private:
    void emitMessage(const char *message, size_t size)
    {
        for (LogOutputPtrListType::iterator loIt = m_logoutputList.begin();
             loIt != m_logoutputList.end(); loIt++)
        {
            LogOutputPtrListType::value_type &lo(*loIt);
            lo->emit(message, size);
        }
    }

    LogOutputPtrListType m_logoutputList;
};

class Log
{
    typedef std::map<std::string, LogCategory> CategoriesMapType;

public:
    Log()
    {
        m_categoriesMap["deb"] = LogCategory();
        m_categoriesMap["warn"] = LogCategory();
        m_categoriesMap["fatal"] = LogCategory();
    }

    LogCategory& operator()(const std::string &category)
    {
        return m_categoriesMap[category];
    }

    bool registerOutput(const std::string &category,
                        boost::shared_ptr<LogOutput> logOutput)
    {
        CategoriesMapType::iterator foundCat = m_categoriesMap.find(category);
        if ( foundCat != m_categoriesMap.end() )
        {
            foundCat->second.addOutput(logOutput);
            return true;
        }

        return false;
    }

private:
    CategoriesMapType m_categoriesMap;
};


class MutexLogOutput: public LogOutput
{
public:
    void emit(const char *message, size_t size)
    {
        boost::unique_lock<boost::mutex> lock(m_queueMutex);
        emitLocked(message, size);

    }
protected:
    virtual void emitLocked(const char *message, size_t size) = 0;

private:
    boost::mutex m_queueMutex;
};

class StdErrLogOutput: public MutexLogOutput
{
public:
    void emitLocked(const char *message, size_t size)
    {
        std::cerr<<message;
    }

private:
};

class StreamLogOutput: public MutexLogOutput
{
public:
    StreamLogOutput(std::ostream &stream)
        : m_stream(stream)
    { }

protected:
    void emitLocked(const char *message, size_t size)
    {
        m_stream<<message;
    }

private:
    std::ostream &m_stream;
};

extern Log g_tdvLog;

void TdvGlobalLogDefaultOutputs();

TDV_NAMESPACE_END

#define TDV_LOG(wn) tdv::g_tdvLog(#wn)

#endif /* TDV_LOG_HPP */
