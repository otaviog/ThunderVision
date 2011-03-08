#include <cuda_runtime.h>
#include <cuda.h>
#include "cuerr.hpp"
#include "cudaconstraits.hpp"

texture<float, 2> imageTex;
const int KernelSize = 49;
const int KernelDim = 7;
const int KernelDimH = KernelDim/2;
const int KernelSizeH = KernelSize/2;

__device__ void mysort(float values[])
{
  for (int i=0; i<KernelSize - 1; i++)
    for (int j=i + 1; j<KernelSize; j++) {
      if ( values[j] < values[i] ) {
        float tmp = values[i];
        values[i] = values[j];
        values[j] = tmp;
      }
    }
}

__global__ void medianKernel(float *dest, const int maxOffset)
{
    const int x = threadIdx.x + blockIdx.x*blockDim.x;
    const int y = threadIdx.y + blockIdx.y*blockDim.y;
    const int offset = x + y*blockDim.x*gridDim.x;
    
    if ( offset < maxOffset ) {        
      int vcnt = 0;
      float values[KernelSize];
      
      for (int i=-KernelDimH; i<=KernelDimH; i++) {
        for (int j=-KernelDimH; j<=KernelDimH; j++) {     
          values[vcnt++] = tex2D(imageTex, x + j, y + i);          
        }
      }
      
      mysort(values);
      
      dest[offset] = values[KernelSizeH];
    }
}

TDV_NAMESPACE_BEGIN

void DevMedianFilterRun(const Dim &dim, float *input_d, float *output_d)
{
  CUerrExp cuerr;  
  
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  cuerr << cudaBindTexture2D(NULL, imageTex, input_d, desc, dim.width(),
                             dim.height(), dim.width()*sizeof(float));
  
  imageTex.addressMode[0] = cudaAddressModeWrap;
  imageTex.addressMode[1] = cudaAddressModeWrap;  
  imageTex.normalized = false;
  imageTex.filterMode = cudaFilterModePoint;
  
  WorkSize ws = CudaConstraits().imageWorkSize(dim);
  medianKernel<<<ws.blocks, ws.threads>>>(output_d, dim.size());

  CUerrExp::checkGlobalError();
  
  cuerr << cudaUnbindTexture(imageTex);
}

TDV_NAMESPACE_END
