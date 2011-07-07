#include "fastwtamatcher.hpp"

TDV_NAMESPACE_BEGIN

FastWTAMatcher::FastWTAMatcher(int maxDisparity)
{
    m_lrpipe = m_rrpipe = NULL;
    m_procs[0] = this;
    m_procs[1] = NULL;
    m_maxDisparity = maxDisparity;
}

void FastWTAMatcher::inputs(ReadPipe<FloatImage> *leftInput,
                            ReadPipe<FloatImage> *rightInput)
{
    m_lrpipe = leftInput;
    m_rrpipe = rightInput;        
}

void FastWTAMatcher::process()
{
    cudaSetDevice(0);
    
    while ( update() )
    {        
    }
}

void FastWTADevRun(Dim dsiDim, float *leftImg_d, float *rightImg_d, 
                   float *dispImg);

bool FastWTAMatcher::update()
{
    WriteGuard<ReadWritePipe<FloatImage, FloatImage> > wguard(m_wpipe);
    FloatImage leftImg, rightImg;   
    if ( m_lrpipe->read(&leftImg) && m_rrpipe->read(&rightImg) )
    {
        float *leftImg_d = leftImg.devMem();
        float *rightImg_d = rightImg.devMem();   
        
        Dim dsiDim(leftImg.dim().width(), leftImg.dim().height(), 
                   m_maxDisparity);
        FloatImage image = FloatImage::CreateDev(
            Dim(dsiDim.width(), dsiDim.height()));
        FastWTADevRun(dsiDim, leftImg_d, rightImg_d, image.devMem());
        
        image.cpuMem();
        wguard.write(image);        
    }
    
    return wguard.wasWrite();
}

TDV_NAMESPACE_END
