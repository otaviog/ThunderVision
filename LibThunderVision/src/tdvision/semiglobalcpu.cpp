#include <boost/scoped_array.hpp>
#include <cassert>
#include "cuerr.hpp"
#include "semiglobalcpu.hpp"

#include <highgui.h>

TDV_NAMESPACE_BEGIN

inline size_t dsiOffset(
    const Dim &dim, size_t x, size_t y, size_t z)
{
    return z + dim.depth()*y + dim.height()*dim.depth()*x;
}

inline float dsiValue(const Dim &dim, const float *mem, size_t x, size_t y, size_t z)
{
    const size_t offset = dsiOffset(dim, x, y, z);
    if ( offset < dim.size() )
    {
        return mem[offset];
    }

    return 0.0f;
}

static void pointMulMtx(const int tmatrix[3][3], int x, int y, int *rX, int *rY)
{
    *rX = x*tmatrix[0][0] + y*tmatrix[0][1] + tmatrix[0][2];
    *rY = x*tmatrix[1][0] + y*tmatrix[1][1] + tmatrix[1][2];
}

const float P1 = 30.0f/255.0f;
const float P2 = 150.0f/255.0f;

#define min4(a, b, c, d) std::min(a, std::min(b, std::min(c, d)))

static void zeroVolume(const Dim &dim,
                       float *vol)
{
    for (size_t y=0; y<dim.height(); y++)
    {
        for (size_t x=0; x<dim.width(); x++)
        {
            for (size_t d=0; d<dim.depth(); d++)
            {
                vol[dsiOffset(dim, y, x, d)] = 0.0f;
            }
        }
    }
}

static float noinf(float v)
{
    //if ( std::isinf(v) )
    //return 0.0f;
    return v;
}

static void costVolume(const Dim &dsiDim,
                       const Dim &scanDim,
                       const int tmatrix[3][3],
                       const float *dsi,
                       float *aggregVol,
                       const float *img1, const float *img2)
{   
    const size_t costStep = dsiDim.depth();
    boost::scoped_array<float> lastCost(new float[costStep*scanDim.height()]);
    boost::scoped_array<float> newCost(new float[costStep*scanDim.height()]);
    
    for ( int y=0; y < static_cast<int>(scanDim.height()); y++)
    {
        int tX, tY;
        pointMulMtx(tmatrix, 0, y, &tX, &tY);

        assert(tX < scanDim.width());
        assert(tY < scanDim.height());

        for (int d=0; d<static_cast<int>(dsiDim.depth()); d++)
        {
            const size_t dsiOff = dsiOffset(dsiDim, tX, tY, d);
            assert(dsiOff < dsiDim.size());

            const float cost = noinf(dsi[dsiOff]);

            lastCost[costStep*y + d] = cost;
            aggregVol[dsiOff] = cost;
        }
    }

    for (int x=1; x < static_cast<int>(scanDim.width()); x++)
    {
        float lastIntensity = 0.0;
        for (int y=0; y < static_cast<int>(scanDim.height()); y++)
        {
            int tX, tY;
            pointMulMtx(tmatrix, x, y, &tX, &tY);

            assert(tX < scanDim.width());
            assert(tY < scanDim.height());

            float minDisp = std::numeric_limits<float>::infinity();
            for (size_t i=0; i<dsiDim.depth(); i++)
            {
                minDisp = std::min(lastCost[costStep*y + i], minDisp);
            }
            //minDisp = lastCost[costStep*y + random()%dsiDim.depth()];
            
            const float P2Adjust = P2/
                std::abs(img1[tY*dsiDim.width() + tX] - lastIntensity); 
            
            for (int d=0; d<static_cast<int>(dsiDim.depth()); d++)
            {
                const size_t dsiOff = dsiOffset(dsiDim, tX, tY, d);
                assert(dsiOff < dsiDim.size());

                const float cost = noinf(dsi[dsiOff]);

                const float lcDm1 = d > 0 
                    ? lastCost[costStep*y + d - 1] 
                    : std::numeric_limits<float>::infinity();
                
                const float lcDp1 = d < static_cast<int>((dsiDim.depth() - 1)) 
                    ? lastCost[costStep*y + d + 1] 
                    : std::numeric_limits<float>::infinity();
                
                const float Lr = cost
                    + min4(lastCost[costStep*y + d],
                           lcDm1 + P1,
                           lcDp1 + P1,
                           minDisp + P2Adjust) - minDisp;

                aggregVol[dsiOff] += Lr;
                newCost[costStep*y + d] = Lr;
            }

            lastIntensity = img1[tY*dsiDim.width() + tX];
        }

        for (int y=0; y < static_cast<int>(scanDim.height()); y++)
        {
            int tX, tY;
            pointMulMtx(tmatrix, x, y, &tX, &tY);

            assert(tX < scanDim.width());
            assert(tY < scanDim.height());
            
            memcpy(lastCost.get(), newCost.get(), 
                   sizeof(float)*scanDim.height()*costStep);
        }
    }
}

static void wta(const Dim &dsiDim,
                const float *dsi,
                float *dispImg)
{
    for (size_t y=0; y<dsiDim.height(); y++)
    {
        for (size_t x=0; x<dsiDim.width(); x++)
        {
            float minAggreg = std::numeric_limits<float>::infinity();
            int minDisp = 0;

            for (size_t d=0; d<dsiDim.depth(); d++)
            {
                const size_t dsiOff = dsiOffset(dsiDim, x, y, d);
                const float value = dsi[dsiOff];

                if ( value < minAggreg )
                {
                    minAggreg = value;
                    minDisp = d;
                }
            }

            dispImg[dsiDim.width()*y + x] = 
                float(minDisp)/float(dsiDim.depth());
        }
    }
}

void SemiGlobalCPU::updateImpl(DSIMem mem, FloatImage img)
{
#if 0
    CUerrExp cuerr;
    const Dim &dim = mem.dim();
    boost::scoped_array<float> dsi(new float[dim.size()]);
    
    cuerr << cudaMemcpy(dsi.get(), mem.mem(), sizeof(float)*dim.size(),
                        cudaMemcpyDeviceToHost);

    boost::scoped_array<float> aggreg(new float[dim.size()]);

    const int fowardM[3][3] = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    };

    const int backwardM[3][3] = {
        {-1, 0, dim.width() - 1},
        {0, 1, 0},
        {0, 0, 1}
    };

    const int topBottomM[3][3] = {
        {0, 1, 0},
        {1, 0, 0},
        {0, 0, 1}
    };

    const int bottomTopM[3][3] = {
        {0, -1, dim.height() - 1},
        {1, 0, 0},
        {0, 0, 1}
    };

    FloatImage left(cvLoadImage("../../res/tsukuba512_L.png")), 
    //FloatImage left(cvLoadImage("rt.png")), 
    //FloatImage left(cvLoadImage("q_left.png")), 
        right(cvLoadImage("../../res/tsukuba512_R.png"));
    zeroVolume(dim, aggreg.get());

    costVolume(dim, tdv::Dim(dim.width(), dim.height()),
               fowardM, dsi.get(), aggreg.get(), 
               left.cpuMem()->data.fl,
               right.cpuMem()->data.fl);
#if 1
    costVolume(dim, tdv::Dim(dim.width(), dim.height()),
               backwardM, dsi.get(), aggreg.get(),
               left.cpuMem()->data.fl,
               right.cpuMem()->data.fl);


    costVolume(dim, tdv::Dim(dim.height(), dim.width()),
               topBottomM, dsi.get(), aggreg.get(),
               left.cpuMem()->data.fl,
               right.cpuMem()->data.fl);

    costVolume(dim, tdv::Dim(dim.height(), dim.width()),
               bottomTopM, dsi.get(), aggreg.get(),
               left.cpuMem()->data.fl,
               right.cpuMem()->data.fl);
#endif

    wta(dim, aggreg.get(), img.cpuMem()->data.fl);
#endif
}

TDV_NAMESPACE_END
