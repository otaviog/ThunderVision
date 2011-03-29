#ifndef TDV_PROCESSGROUP_HPP
#define TDV_PROCESSGROUP_HPP

#include <vector>
#include <tdvbasic/common.hpp>

TDV_NAMESPACE_BEGIN

class Process;

class ProcessGroup
{
public:        
    virtual Process** processes() = 0;
    
private:

};

class ArrayProcessGroup: public ProcessGroup
{
public:
    ArrayProcessGroup()
    {
        m_procs.push_back(NULL);
    }
    
    ArrayProcessGroup(Process **procs)
    {
        m_procs.push_back(NULL);
        addProcess(procs);
    }
    
    ArrayProcessGroup(ProcessGroup &procs)
    {
        m_procs.push_back(NULL);
        addProcess(procs);
    }

    virtual Process** processes()
    {
        return &m_procs[0];
    }
    
    void addProcess(Process *proc)
    {
        m_procs.insert(m_procs.end() - 1, proc);
    }
    
    void addProcess(ProcessGroup &group)
    {
        addProcess(group.processes());
    }
    
    void addProcess(Process **procs);
        
private:
    std::vector<Process*> m_procs;
};

TDV_NAMESPACE_END

#endif /* TDV_PROCESSGROUP_HPP */
