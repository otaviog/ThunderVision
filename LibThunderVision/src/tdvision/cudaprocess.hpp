#ifndef TDV_CUDAPROCESS_HPP
#define TDV_CUDAPROCESS_HPP

#include <vector>
#include <tdvbasic/common.hpp>
#include "process.hpp"

TDV_NAMESPACE_BEGIN

class WorkUnit;

class CUDAProcess: public Process
{
public:
    CUDAProcess(int deviceId)
    {
        m_deviceId = deviceId;
        m_end = false;
    }
    
    void process();
    
    void finish();

    void addWork(WorkUnit *wu)
    {
        m_units.push_back(wu);
    }
        
private:
    std::vector<WorkUnit*> m_units;    
    int m_deviceId;
    
    bool m_end;
};

TDV_NAMESPACE_END

#endif /* TDV_CUDAPROCESS_HPP */
