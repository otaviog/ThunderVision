#include "sink.hpp"
#include "reprojectprocess.hpp"

TDV_NAMESPACE_BEGIN

void ReprojectProcess::process()
{        
    FloatImage disp;    
    CvMat *origin;
    
    while ( m_originPipe->read(&origin) && m_dispPipe->read(&disp) )
    {
        if ( m_reproj != NULL )
        {
            assert(m_proj != NULL);
            m_reproj->reproject(disp, origin, m_proj);
        }
        CvMatSinkPol::sink(origin);
    }
}


TDV_NAMESPACE_END
