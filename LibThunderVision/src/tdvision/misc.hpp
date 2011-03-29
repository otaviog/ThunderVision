#ifndef TDV_MISC_HPP
#define TDV_MISC_HPP

#include "workunit.hpp"
#include "pipe.hpp"

namespace misc
{
    template<typename InType, typename OutType = InType>
    class MonoWorkUnit: public WorkUnit
    {
    public:
        void input(ReadPipe<InType> *rpipe)
        {
            m_rpipe = rpipe;
        }
        
        ReadPipe<OutType>* output()
        {
            return &m_wpipe;
        }
        
    private:
        ReadPipe<InType> *m_rpipe;
        ReadWritePipe<OutType> m_wpipe;
    };

    template<typename InType, typename OutType = InType>
    class StereoWorkUnit: public WorkUnit
    {
    public:
        StereoWorkUnit()
        {
            m_lrpipe = NULL;
            m_rrpipe = NULL;
        }
        
        void inputs(ReadPipe<InType> *lrpipe, ReadPipe<InType> *rrpipe)
        {
            m_rrpipe = rrpipe;
            m_lrpipe = lrpipe;
        }
        
        ReadPipe<OutType>* output()
        {
            return &m_wpipe;
        }
    private:
        ReadPipe<InType> *m_lrpipe, *m_rrpipe;
        ReadWritePipe<OutType> m_wpipe;

    };
}

#endif /* TDV_MISC_HPP */
