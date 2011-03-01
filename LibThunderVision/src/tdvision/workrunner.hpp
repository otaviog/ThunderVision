#ifndef TDV_WORKRUNNER_HPP
#define TDV_WORKRUNNER_HPP

#include <tdvbasic/common.hpp>

TDV_NAMESPACE_BEGIN

class WorkRunner
{
public:
    WorkRunner();
    
    virtual ~WorkRunner();
    
    template<typename Work1, typename Work2>
    void connect(Work1 *w1, Work2 *w2)
    {
        
    }
    
    void run();
    
private:    
    void errorOcurred(const std::exception &ex);

    std::set<WorkUnit*> m_wunits;
    std::set<BaseWritePipe*> m_pipes;
    
};
TDV_NAMESPACE_END

#endif /* TDV_WORKRUNNER_HPP */
