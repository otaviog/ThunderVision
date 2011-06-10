#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda.h>
#include "cuerr.hpp"
#include "cudaconstraits.hpp"
#include "dsimemutil.h"

texture<float, 2> texLeftImg;
texture<float, 2> texRightImg;

#define SSD_WIND_DIM 5
#define SSD_WIND_START -3
#define SSD_WIND_END 4

struct DSIDim
{
  int x, y, z;
};

#if 1
inline __host__ __device__ uint dsiOffset(const DSIDim dim, uint x, uint y, uint z)
{
    return dim.z*dim.y*x + dim.z*y + z;
}
inline __host__ __device__ void dsiSetIntensity(const DSIDim dim, uint x, uint y, uint z, float value, 
    float *dsi)
{
  dsi[dsiOffset(dim, x, y, z)] = value;
}
#else
#define dsiOffset(dim, px, py, pz) dim.z*dim.y*px + dim.z*py + pz
#define dsiSetIntensity(dim, x, y, z, value, dsi) dsi[dsiOffset(dim, x, y, z)] = value
#endif

__device__ float ssdAtDisp(int x, int y, int disp)
{
  float sum = 0.0f;

  for (int row=SSD_WIND_START; row<SSD_WIND_END; row++)
    for (int col=SSD_WIND_START; col<SSD_WIND_END; col++) {

      float lI = tex2D(texLeftImg, x + col, y + row),
        rI = tex2D(texRightImg, x + col - disp, y + row);

      sum += (lI - rI)*(lI - rI);
    }

  return sum;
}


#if 1
__global__ void ssdKern(const dim3 dsiDim, cudaPitchedPtr dsiMem)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y; 
  
  if ( x < dsiDim.x && y < dsiDim.y ) {
    float *dsiRow = (float*) (((char *) dsiMem.ptr) + dsiMem.pitch*dsiDim.x*y
                              + dsiMem.pitch*x);

    for (int disp=0; disp < dsiDim.z; disp++) {
      float ssdValue = CUDART_INF_F;

      if ( x - disp >= 0 ) {
        ssdValue = ssdAtDisp(x, y, disp);
      }
      
      //dsiSetIntensity(dsiDim, x, y, disp, ssdValue, (float*) dsiMem.ptr);
      dsiRow[disp] = ssdValue;
    }
  }
}
#else
__global__ void ssdKern(const DSIDim dsiDim, const int maxDisparity, 
                        float *dsiMem)
{  
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;     

  if ( x < dsiDim.x && y < dsiDim.y ) {
    
    for (int disp=0; (disp < maxDisparity); disp++) {   
      float ssdValue = CUDART_INF_F; 
      if ( x - disp >= 0 ) {       
        ssdValue = ssdAtDisp(x, y, disp);                    
      }
      
      dsiSetIntensity(dsiDim, x, y, disp, ssdValue, dsiMem);
    }    
    
  }
}
#endif

TDV_NAMESPACE_BEGIN

void SSDDevRun(Dim dsiDim, float *leftImg_d, float *rightImg_d,
               cudaPitchedPtr dsiMem)
{
  CUerrExp err;

  err << cudaBindTexture2D(NULL, texLeftImg, leftImg_d,
                           cudaCreateChannelDesc<float>(),
                           dsiDim.width(), dsiDim.height(),
                           dsiDim.width()*sizeof(float));

  err << cudaBindTexture2D(NULL, texRightImg, rightImg_d,
                           cudaCreateChannelDesc<float>(),
                           dsiDim.width(), dsiDim.height(),
                           dsiDim.width()*sizeof(float));

  texLeftImg.addressMode[0] = texRightImg.addressMode[0] = cudaAddressModeWrap;
  texLeftImg.addressMode[1] = texRightImg.addressMode[1] = cudaAddressModeWrap;
  texLeftImg.normalized = texRightImg.normalized = false;
  texLeftImg.filterMode = texRightImg.filterMode = cudaFilterModePoint;

  CudaConstraits constraits;
  WorkSize ws = constraits.imageWorkSize(dsiDim);

  DSIDim ddim;
  ddim.x = dsiDim.width();
  ddim.y = dsiDim.height();
  ddim.z = dsiDim.depth();
  
  ssdKern<<<ws.blocks, ws.threads>>>(tdvDimTo(dsiDim), dsiMem);
  //ssdKern<<<ws.blocks, ws.threads>>>(ddim, dsiDim.depth(), (float*) dsiMem.ptr);
  
  err << cudaThreadSynchronize();
}

TDV_NAMESPACE_END