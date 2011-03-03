#include "ssdwudev.hpp"


TDV_NAMESPACE_BEGIN

void DevSSDRun(int maxDisparity, 
               Dim imgDim, float *leftImg, float *rightImg,
               Dim dsiDim, float *dsiMem);

SSDWUDev::SSDWUDev(int disparityMax, size_t memoryByPacket)
{
    workName("SSD on device");
}

void SSDWUDev::process()
{    
    FloatImage leftImg;
    FloatImage rightImg;
    
    while ( m_lrpipe->read(&leftImg) && m_rrpipe->read(&rightImg) )
    {            
        const size_t width = std::min(leftImg.dim().width(), rightImg.dim().height());
        const size_t height = std::min(leftImg.dim().height(), rightImg.dim().height());
        const size_t depth = m_maxDisparaty;
        
        const size_t totalSize = width*height*depth;
        const size_t numPackets = (totalSize + m_memoryByPacket - 1)/m_memoryByPacket;
        const size_t rowsByPacket = height/numPackets;
                
        for (size_t i=0; i<numPackets; i++)
        {            
            const size_t rowCount = rowsByPacket*i;            
            const size_t packetHeight = (rowCount <= height) 
                ? rowsByPacket
                : (rowCount - height);
            
            const Dim pktDim(width, packetHeight, depth);
            DSIMem dsi = DSIMem::Create(pktDim);            
            
            float *leftImg_d = leftImg.devMem();
            float *rightImg_d = leftImg.devMem();
            
            DevSSDRun(m_maxDisparaty, Dim(width, height), 
                      leftImg_d, rightImg_d, pktDim, dsi.mem());
            
            m_wpipe.write(dsi);
        }
    }
    
    m_wpipe.finish();
}

TDV_NAMESPACE_END
