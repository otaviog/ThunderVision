#include <iostream>

#include <cmath>
#include "mutualinformationcpu.hpp"
#include "dim.hpp"

TDV_NAMESPACE_BEGIN

inline int valueToHIdx(float value, size_t sz)
{
    int idx = int(value*sz);
    if ( idx == sz )
        return idx - 1;
    
    return idx;
}

inline size_t dsiOffset(
    const Dim &dim, size_t x, size_t y, size_t z)
{
    return z + dim.depth()*y + dim.height()*dim.depth()*x;
}

MutualInformationCPU::MutualInformationCPU(int disparityMax)
{
    m_lrpipe = NULL;
    m_rrpipe = NULL;
    m_maxDisparaty = disparityMax;
}

float MutualInformationCPU::mutualInformation(const float A[], const float B[], size_t sz, 
                                              int histA[], int histB[], int histAB[], size_t histSize)
{
    float mi = 0.0f;

    for (size_t i=0; i<histSize; i++)
    {
        histA[i] = 0;
        histB[i] = 0;
        
        for ( size_t j=0; j<histSize; j++)
        {
            histAB[i*histSize + j] = 0;
        }
    }

    for (size_t i=0; i<sz; i++)
    {
        histA[valueToHIdx(A[i], histSize)] += 1;
        histB[valueToHIdx(B[i], histSize)] += 1;
    }

    for (size_t i=0; i<sz; i++)
    {
        const int histIdxI = valueToHIdx(A[i], histSize);
        for (size_t j=0; j<sz; j++)
        {
            const int histIdxJ = valueToHIdx(B[i], histSize);
            
            histAB[histIdxI*histSize + histIdxJ] += 1;
        }
    }
    
    for (size_t i=0; i<sz; i++)
    {
        const int histIdxI = valueToHIdx(A[i], histSize);
        const float probI = histA[histIdxI];
        
        for (size_t j=0; j<sz; j++)
        {
            const int histIdxJ = valueToHIdx(B[i], histSize);            
            const float probJ = histB[histIdxJ];            
            const float probIJ = histAB[histIdxI*histSize + histIdxJ];
            
            mi += probIJ*log(probIJ/(probI*probJ));
        }
    }
        
    return mi;
}

#define SAMPLES 9
#define HIST_SIZE 255

void MutualInformationCPU::calcDSI(FloatImage limg_d, FloatImage rimg_d, float **dsiRet, Dim *dimRet)
{
    float *limg = limg_d.cpuMem()->data.fl;
    float *rimg = rimg_d.cpuMem()->data.fl;

    const size_t width = limg_d.dim().width();
    const size_t height = limg_d.dim().height();
    const size_t size = width*height;
    
    Dim dsiDim(width, height, m_maxDisparaty);
    float *dsi = new float[dsiDim.size()];
    
    for (int i=0; i<static_cast<int>(width); i++)
    {
        for (int j=0; j<static_cast<int>(height); j++)
        {            
            float A[SAMPLES];
            float B[SAMPLES];
            int histA[HIST_SIZE];
            int histB[HIST_SIZE];
            int histAB[HIST_SIZE*HIST_SIZE];
            
            int countA = 0;                        
            for ( int r=i - 1; r<i + 2; r++)
            {
                for (int c=j - 1; c < j + 2; c++)
                {
                    const size_t idxA = r*width + c;
                    if ( idxA < size )
                    {
                        A[countA++] = limg[idxA];
                    }
                }
            }
            
            for ( size_t disp=0; disp<m_maxDisparaty; disp++)
            {                          
                int countB = 0;
                for ( int r=i - 1; r<i + 2; r++)
                {
                    for (int c=j - 1; c < j + 2; c++)
                    {
                        const size_t idxB = r*width + c - disp;
                        if ( idxB < size )
                        {
                            B[countB++] = rimg[idxB];
                        }
                    }
                }
                
                const float mi = mutualInformation(A, B, std::min(countA, countB), 
                                                   histA, histB, histAB, HIST_SIZE);
                dsi[dsiOffset(dsiDim, j, i, disp)] = mi;
            }            
            
        }
        
        std::cout<<"Done "<<float(i)/float(height)*100.0f<<"%"<<std::endl;
    }
    
    *dsiRet = dsi;
    *dimRet = dsiDim;
}

bool MutualInformationCPU::update()
{
    WriteGuard<ReadWritePipe<DSIMem, DSIMem> > wguard(m_wpipe);
    FloatImage leftImg;
    FloatImage rightImg;

    if ( m_lrpipe->read(&leftImg) && m_rrpipe->read(&rightImg) )
    {
        
        float *dsiMem;
        Dim dsiDim(0);
        
        calcDSI(leftImg, rightImg, &dsiMem, &dsiDim);
        
        DSIMem dsi = DSIMem::Create(dsiDim, leftImg);
        cudaMemcpy(dsi.mem(), dsiMem, dsiDim.size(), cudaMemcpyHostToDevice);
        
        wguard.write(dsi);
    }

    return wguard.wasWrite();

}

TDV_NAMESPACE_END
