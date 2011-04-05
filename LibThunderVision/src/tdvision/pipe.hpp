#ifndef TDV_PIPE_HPP
#define TDV_PIPE_HPP

#include <queue>
#include <tdvbasic/common.hpp>
#include <boost/thread.hpp>

TDV_NAMESPACE_BEGIN

class BaseReadPipe
{
public:
    virtual ~BaseReadPipe()
    {
    }
    
    virtual void waitRead() = 0;    
};

template<typename ReadType>
class ReadPipe: public BaseReadPipe
{
public:        
    virtual ~ReadPipe()
    { }
    
    virtual bool read(ReadType *outvalue) = 0;
};

class BaseWritePipe
{
public:
    virtual ~BaseWritePipe()
    {
    }

    virtual void finish() = 0;
};

template<typename WriteType>
class WritePipe: public BaseWritePipe
{
public:    
    typedef WriteType WriteValueType;
    
    virtual ~WritePipe()
    { }
    
    virtual void write(WriteType packet) = 0;    
};

template<typename Type>
class PasstruPipeAdapter
{
public:
    static Type adapt(Type value)
    {
        return value;
    }
private:
};

template<typename In, typename Out>
class CastPipeAdapter
{
public:
    static Out adapt(In value)
    {
        return static_cast<Out>(value);
    }
    
private:
};

template<typename ReadType, typename WriteType = ReadType, 
         typename Adapter = PasstruPipeAdapter<ReadType> >
class ReadWritePipe: public WritePipe<WriteType>, public ReadPipe<ReadType>
{
public:
    ReadWritePipe(size_t maxSize = 100)
    {
        m_end = false;
        m_maxSize = maxSize;
    }
    
    void write(WriteType value);    

    bool read(ReadType *outvalue);
        
    void finish()
    {
        boost::mutex::scoped_lock lock(m_queueMutex);
        m_end = true;
        m_queueCond.notify_one();
    }
    
    void waitRead()
    {
        ReadType tp;
        (void) read(&tp);
    }
    
private:
    std::queue<ReadType> m_queue;
    boost::mutex m_queueMutex;
    boost::condition_variable m_queueCond, m_enqueueCond;
    bool m_end;
    size_t m_maxSize;
};

template<typename ReadType, typename WriteType, typename Adapter>
void ReadWritePipe<ReadType, WriteType, Adapter>::write(WriteType value)
{
    boost::mutex::scoped_lock lock(m_queueMutex);
    while (m_queue.size() >= m_maxSize && !m_end )
    {
        m_enqueueCond.wait(lock);
    }
    
    if ( !m_end )
    {
        m_queue.push(Adapter::adapt(value));    
        m_queueCond.notify_one();    
    }    
}

template<typename ReadType, typename WriteType, typename Adapter>
bool ReadWritePipe<ReadType, WriteType, Adapter>::read(ReadType *outread)
{
    boost::mutex::scoped_lock lock(m_queueMutex);
    while ( m_queue.empty() && !m_end )
    {
        m_queueCond.wait(lock);
    }

    bool hasEmpty = m_queue.empty();
    if ( !hasEmpty )
    {
        *outread = m_queue.front();
        m_queue.pop();
    }
    
    m_enqueueCond.notify_one();        
    
    return !hasEmpty;
}

template<typename WritePipeType>
class WriteGuard: public  WritePipe<typename WritePipeType::WriteValueType>
{
public:
    typedef typename WritePipeType::WriteValueType WriteValueType;
    
    WriteGuard(WritePipeType &pipe)
        : m_pipe(pipe)
    { m_wasWrite = false; }
    
    ~WriteGuard()
    {
        if ( !m_wasWrite )
        {
            m_pipe.finish();
        }
    }
    
    void write(WriteValueType packet)
    {
        m_pipe.write(packet);
        m_wasWrite = true;
    }
    
    void finish()
    {
        m_pipe.finish();
    }
    
    bool wasWrite() const
    {
        return m_wasWrite;
    }

private:
    WritePipeType &m_pipe;
    bool m_wasWrite;
};

TDV_NAMESPACE_END

#endif /* TDV_PIPE_HPP */
