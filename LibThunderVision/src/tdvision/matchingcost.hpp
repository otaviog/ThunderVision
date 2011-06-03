#ifndef TDV_MATCHINGCOST_HPP
#define TDV_MATCHINGCOST_HPP

#include <tdvbasic/common.hpp>
#include "workunit.hpp"
#include "pipe.hpp"
#include "dsimem.hpp"
#include "floatimage.hpp"
#include "benchmark.hpp"

TDV_NAMESPACE_BEGIN

class MatchingCost: public WorkUnit
{
public:
    virtual void inputs(ReadPipe<FloatImage> *leftInput,
                        ReadPipe<FloatImage> *rightInput) = 0;
    
    virtual ReadPipe<DSIMem>* output() = 0;
    
    virtual Benchmark benchmark() const = 0;
};

class AbstractMatchingCost: public MatchingCost
{
public:
    AbstractMatchingCost(int disparityMax);
    
    void inputs(ReadPipe<FloatImage> *lpipe, ReadPipe<FloatImage> *rpipe)
    {
        m_lrpipe = lpipe;
        m_rrpipe = rpipe;
    }
    
    ReadPipe<DSIMem>* output()
    {
        return &m_wpipe;
    }
    
    bool update();        
    
    virtual Benchmark benchmark() const
    {
        return m_mark;
    }
    
protected:
    virtual void updateImpl(FloatImage left, FloatImage right,
                            DSIMem mem) = 0;
    
private:
    ReadPipe<FloatImage> *m_lrpipe, *m_rrpipe;
    ReadWritePipe<DSIMem, DSIMem> m_wpipe;
    size_t m_maxDisparaty;
    DSIMem m_dsi;
    Benchmark m_mark;
};

TDV_NAMESPACE_END

#endif /* TDV_MATCHINGCOST_HPP */
