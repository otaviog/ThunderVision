#include "semigloballabmatcher.hpp"

TDV_NAMESPACE_BEGIN

SemiglobalLabMatcher::SemiglobalLabMatcher(int maxDisparity)
{
    m_lrpipe = m_rrpipe = NULL;
    m_procs[0] = this;
    m_procs[1] = NULL;
    m_maxDisparity = maxDisparity;
    m_zeroAggregDSI = true;
}

void SemiglobalLabMatcher::inputs(ReadPipe<FloatImage> *leftInput,
                                  ReadPipe<FloatImage> *rightInput)
{
    m_lrpipe = leftInput;
    m_rrpipe = rightInput;        
}

void SemiglobalLabMatcher::process()
{
    cudaSetDevice(0);
    
    while ( update() )
    {        
    }
    
    m_aggregDSI.unalloc();
    m_sgPaths.unalloc();
    m_zeroAggregDSI = true;
}

void SemiGlobalLabDevRun(const Dim &dsiDim,
                         const SGPath *pathsArray, size_t pathCount,
                         float *leftImg_d, float *rightImg_d, 
                         cudaPitchedPtr aggregDSI,
                         float *dispImg, bool zeroAggregDSI);

bool SemiglobalLabMatcher::update()
{
    WriteGuard<ReadWritePipe<FloatImage, FloatImage> > wguard(m_wpipe);
    FloatImage leftImg, rightImg;   
    if ( m_lrpipe->read(&leftImg) && m_rrpipe->read(&rightImg) )
    {
        Dim dsiDim(leftImg.dim().width(), leftImg.dim().height(), 
                   m_maxDisparity);
        
        float *leftImg_d = leftImg.devMem();
        float *rightImg_d = rightImg.devMem();   
        FloatImage dispImage = FloatImage::CreateDev(
            Dim(dsiDim.width(), dsiDim.height()));
     
        cudaPitchedPtr aggregDSI = m_aggregDSI.mem(dsiDim);
        SGPath *paths = m_sgPaths.getDescDev(dispImage.dim());
                   
        
        SemiGlobalLabDevRun(dsiDim, paths, m_sgPaths.pathCount(),
                            leftImg_d, rightImg_d,
                            aggregDSI, dispImage.devMem(), m_zeroAggregDSI);
        m_zeroAggregDSI = false;
        
        dispImage.cpuMem();
        wguard.write(dispImage);        
    }
    
    return wguard.wasWrite();
}

TDV_NAMESPACE_END
