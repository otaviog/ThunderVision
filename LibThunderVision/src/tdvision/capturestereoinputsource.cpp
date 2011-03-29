#include "capturestereoinputsource.hpp"

TDV_NAMESPACE_BEGIN

CaptureStereoInputSource::CaptureStereoInputSource()
{
    m_procs[0] = this;
    m_procs[1] = NULL;    
    m_stopCap = false;
}

void CaptureStereoInputSource::init()
{
    m_capture1.init(0);
    m_capture2.init(1);
}

void CaptureStereoInputSource::init(const std::string &filename1, const std::string &filename2)
{
    m_capture1.init(filename1);
    m_capture2.init(filename2);
}

void CaptureStereoInputSource::process()
{
    try
    {
        while ( !m_stopCap ) 
        {
            m_capture1.update();
            m_capture2.update();
        }
        
        m_capture1.dispose();
        m_capture2.dispose();
    }
    catch (const std::exception &ex )
    {        
        m_capture1.dispose();
        m_capture2.dispose();
        throw ex;
    }
}

TDV_NAMESPACE_END
