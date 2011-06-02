#ifndef TDV_PIPE_HPP
#define TDV_PIPE_HPP

#include <queue>
#include <tdvbasic/common.hpp>
#include <boost/thread.hpp>

TDV_NAMESPACE_BEGIN

/**
 * Base interface for a pipe with read job.
 * 
 * @author Otavio Gomes <otaviolmiro@gmail.com>
 */
class BaseReadPipe
{
public:
    virtual ~BaseReadPipe()
    {
    }
    
    /**
     * Waits a read and discards the received data.
     */
    virtual void waitRead() = 0;    
};

/**
 * Typed interface for read pipes. This interface is
 * the one to be used to specify a abstract 
 * communication between another work unit.
 * @author Otavio Gomes <otaviolmiro@gmail.com>
 */
template<typename ReadType>
class ReadPipe: public BaseReadPipe
{
public:        
    virtual ~ReadPipe()
    { }
    
    /**
     * Wait until read a new element.
     * @param outvalue returns the read element.
     * @return true if a element was read; false if this
     * pipe has no more elements, and the process reading 
     * may exit.
     */
    virtual bool read(ReadType *outvalue) = 0;
};

/**
 * Base interface for write pipe, contains basic controll mechanism.
 * @author Otavio Gomes <otaviolmiro@gmail.com>
 */
class BaseWritePipe
{
public:
    virtual ~BaseWritePipe()
    {
    }

    /**
     * Writes that no more data will be writren.
     */
    virtual void finish() = 0;
};

/**
 * Typed interface for write pipe. Gives a common way to write
 * elements.
 */
template<typename WriteType>
class WritePipe: public BaseWritePipe
{
public:    
    typedef WriteType WriteValueType;
    
    virtual ~WritePipe()
    { }
    
    /**
     * Writes to the pipe.
     * @param packet element to write.
     */
    virtual void write(WriteType packet) = 0;    
};

/**
 * @decrepted
 */
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

/**
 * @decrepted
 */
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

/**
 * All other pipe classes are abstract. This one is the glue
 * between them. This class combines the read and write
 * pipes to form the common pipe type in thundervision, 
 * it's use the bounded-buffer logic.
 * @author Otavio Gomes <otaviolmiro@gmail.com>
 */
template<typename ReadType, typename WriteType = ReadType, 
         typename Adapter = PasstruPipeAdapter<ReadType> >
class ReadWritePipe: public WritePipe<WriteType>, public ReadPipe<ReadType>
{
public:


    /**
     * Constructor.
     * @param maxSize maximum size for the buffer queue.
     * When the queue reach this size, then the next writes
     * will wait, according to the bounded-buffer logic.
     */
    ReadWritePipe(size_t maxSize = 100)
    {
        m_end = false;
        m_maxSize = maxSize;
    }
    
    ~ReadWritePipe()
    {
        finish();
    }
    
    /**
     * Enqueues a value for futher read.
     */ 
    void write(WriteType value);    

    /**
     * Wait until read a new element.
     * @param outvalue returns the read element.
     * @return true if a element was read; false if this
     * pipe has no more elements, and the process reading 
     * may exit.
     */
    bool read(ReadType *outvalue);
        
    /**
     * Sets in next empty read to return false, indicating
     * the end of data in this pipe.
     */
    void finish()
    {
        boost::mutex::scoped_lock lock(m_queueMutex);
        m_end = true;
        m_queueCond.notify_one();
    }
    
    /**
     * Unsets the end of data in this pipe.
     */
    void reset()
    {
        m_end = false;
    }
    
    /**
     * Discards a read.
     */
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
#if 1
    boost::mutex::scoped_lock lock(m_queueMutex);
    m_end = false;
    
    while (m_queue.size() >= m_maxSize && !m_end )
    {
        m_enqueueCond.wait(lock);
    }
    
    if ( !m_end )
    {
        m_queue.push(Adapter::adapt(value));
        m_queueCond.notify_one();    
    }    
#else
    m_queue.push(Adapter::adapt(value));
    m_queueCond.notify_one();    
#endif
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

/**
 * Works like a scope guard to prevent writing process to
 * forget the finish command on a write pipe.
 */
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
