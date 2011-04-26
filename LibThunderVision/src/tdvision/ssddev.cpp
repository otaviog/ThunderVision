#include "benchmark.hpp"
#include "ssddev.hpp"

TDV_NAMESPACE_BEGIN

void DevSSDRun(int maxDisparity, 
               Dim dsiDim, float *leftImg, float *rightImg,
               float *dsiMem);

SSDDev::SSDDev(int disparityMax)
{
    workName("SSD");
    m_maxDisparaty = disparityMax;
    m_memoryByPacket = 0;
}

#if 0
bool SSDDev::update()
{   
    WriteGuard<ReadWritePipe<DSIMem, DSIMem> > wguard(m_wpipe);
    FloatImage leftImg;
    FloatImage rightImg;
    
    if ( m_lrpipe->read(&leftImg) && m_rrpipe->read(&rightImg) )
    {            
        const size_t width = std::min(leftImg.dim().width(), 
                                      rightImg.dim().width());
        const size_t height = std::min(leftImg.dim().height(), 
                                       rightImg.dim().height());
        const size_t depth = m_maxDisparaty;
        
        const size_t totalSize = width*height*depth;
        const size_t numPackets = 
            (totalSize + m_memoryByPacket - 1)/m_memoryByPacket;
        const size_t rowsByPacket = height/numPackets;
            
        CudaBenchmarker bm;
        bm.begin();
        
        for (size_t i=0; i<numPackets; i++)
        {            
            const size_t rowCount = rowsByPacket*i;            
            const size_t packetHeight = (rowCount <= height) 
                ? rowsByPacket
                : (rowCount - height);
            
            const Dim pktDim(width, packetHeight, depth);
            DSIMem dsi = DSIMem::Create(pktDim);            
            
            float *leftImg_d = leftImg.devMem();
            float *rightImg_d = rightImg.devMem();
                        
            DevSSDRun(m_maxDisparaty, Dim(width, height), 
                      leftImg_d, rightImg_d, pktDim, dsi.mem());
            
            m_wpipe.write(dsi);
        }
        
        bm.end();
    }
    
    m_wpipe.finish();
    return false;
}
#else
bool SSDDev::update()
{
    WriteGuard<ReadWritePipe<DSIMem, DSIMem> > wguard(m_wpipe);
    FloatImage leftImg;
    FloatImage rightImg;
    
    if ( m_lrpipe->read(&leftImg) && m_rrpipe->read(&rightImg) )
    {            
        const size_t width = std::min(leftImg.dim().width(), 
                                      rightImg.dim().width());
        const size_t height = std::min(leftImg.dim().height(), 
                                       rightImg.dim().height());
        const size_t depth = m_maxDisparaty;
                    
        CudaBenchmarker bm;
        bm.begin();               
        const Dim pktDim(width, height, depth);
        DSIMem dsi = DSIMem::Create(pktDim);            
            
        float *leftImg_d = leftImg.devMem();
        float *rightImg_d = rightImg.devMem();
                        
        DevSSDRun(m_maxDisparaty, pktDim, 
                  leftImg_d, rightImg_d, dsi.mem());
            
        m_wpipe.write(dsi);        
        
        bm.end();
    }
    
    return wguard.wasWrite();
}
#endif

TDV_NAMESPACE_END
