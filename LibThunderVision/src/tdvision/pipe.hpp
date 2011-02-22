#ifndef TDV_PIPE_HPP
#define TDV_PIPE_HPP

#include <queue>
#include <tdvbasic/common.hpp>
#include <boost/thread.hpp>

TDV_NAMESPACE_BEGIN

template<typename ReadType>
class ReadPipe
{
public:        
    virtual ~ReadPipe()
    { }

    virtual bool waitPacket() = 0;
    
    virtual ReadType read() = 0;
};

template<typename WriteType>
class WritePipe
{
public:    
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

template<typename ReadType, typename WriteType, 
         typename Adapter = PasstruPipeAdapter<ReadType> >
class ReadWritePipe: public WritePipe<WriteType>, public ReadPipe<ReadType>
{
public:
    ReadWritePipe()
    {
        m_end = false;
    }
    
    virtual void write(WriteType value);
    
    virtual bool waitPacket();

    virtual ReadType read();
        
    void end()
    {
        boost::mutex::scoped_lock lock(m_queueMutex);
        m_end = true;
        m_queueCond.notify_one();
    }
    
private:
    std::queue<ReadType> m_queue;
    boost::mutex m_queueMutex;
    boost::condition_variable m_queueCond;
    bool m_end;
};

template<typename ReadType, typename WriteType, typename Adapter>
void ReadWritePipe<ReadType, WriteType, Adapter>::write(WriteType value)
{
    boost::mutex::scoped_lock lock(m_queueMutex);
    m_queue.push(Adapter::adapt(value));
    
    m_queueCond.notify_one();    
}

template<typename ReadType, typename WriteType, typename Adapter>
bool ReadWritePipe<ReadType, WriteType, Adapter>::waitPacket()
{
    boost::mutex::scoped_lock lock(m_queueMutex);
    // Condition overrun, we must quit and return true or quit and return false?
    while ( m_queue.empty() && !m_end )
    {
        m_queueCond.wait(lock);
    }
    
    return !m_queue.empty();
}

template<typename ReadType, typename WriteType, typename Adapter>
ReadType ReadWritePipe<ReadType, WriteType, Adapter>::read()
{
    boost::mutex::scoped_lock(m_queueMutex);
    ReadType value = m_queue.front();
    m_queue.pop();
    
    return value;
}

TDV_NAMESPACE_END

#endif /* TDV_PIPE_HPP */


