#include "dim.hpp"
#include "dsimemutil.h"

__device__ void dynamicprogSlice(uint y, uint z, const float *costDSI,
                                 float *sumCostDSI, char *pathDSI)
{
  for (uint x=0; x<g_dsiDim.x; x++) {    
      const uint c0Offset = dsiOffset(x, y, z);  
      const float c0 = costDSI[c0Offset];
      
      /**
       * c1\                                     \
       * c2-c0
       * c3/
       */  
      const float c1 = dsiIntensityClamped(x - 1, y, z - 1, sumCostDSI);
      const float c2 = dsiIntensityClamped(x - 1, y, z, sumCostDSI);
      const float c3 = dsiIntensityClamped(x - 1, y, z + 1, sumCostDSI);      
      
      float m;      
      int p;  
      if ( c1 < c2 && c1 < c3 ) {
        m = c1;
        p = 1;
      } else if ( c2 < c3 ) {
        m = c2;
        p = 0;
      } else {
        m = c3;
        p = -1;
      }
      
      sumCostDSI[c0Offset] = c0 + m;
      pathDSI[c0Offset] = p;
  }
}

__global__ void reduceImage(const float *sumCostDSI, const char *pathDSI, 
                            float *dispImg)
{
  const uint y = threadIdx.x + blockIdx.x*blockDim.x; // slice

  uint lastMinZ = 0;
  float min = dsiIntensityClamped(g_dsiDim.x - 1, y, 0, sumCostDSI);
  
  for (uint z=1; z<g_dsiDim.z; z++) {
    const float sc = dsiIntensityClamped(g_dsiDim.x - 1, y, z, sumCostDSI);    
    if ( sc < min ) {
      lastMinZ = z;
      min = sc;
    }          
  }
  
  uint imgOffset = y*g_dsiDim.x + g_dsiDim.x - 1;
  dispImg[imgOffset] = float(lastMinZ)/float(g_dsiDim.z);
    
  for (int x=g_dsiDim.x - 2; x >= 0; x--) {
    const uint offset = dsiOffset(x, y, lastMinZ);
    const uint nz = lastMinZ + pathDSI[offset];
    
    if ( nz < g_dsiMaxOffset )
      lastMinZ = nz;
    
    imgOffset = y*g_dsiDim.x + x;
    dispImg[imgOffset] = float(lastMinZ)/float(g_dsiDim.z);    
  }
}

__global__ void dynamicprog(float *costDSI, float *sumCostDSI, char *pathDSI)
{
  uint z = threadIdx.x + blockIdx.x*blockDim.x;
  uint y = threadIdx.y + blockIdx.y*blockDim.y;
  
  dynamicprogSlice(y, z, costDSI, sumCostDSI, pathDSI);    
}

void runDynamicProg(float *dsi, const tdv::Dim &dsiDim)
{
  dim3 cuda_dsiDim(dsiDim.width(), dsiDim.height(), dsiDim.depth());
  uint maxOffset = dsiDim.size();
  
  cudaMemcpyToSymbol(&g_dsiDim, &cuda_dsiDim, sizeof(dim3));
  cudaMemcpyToSymbol(&g_dsiMaxOffset, &maxOffset, sizeof(uint));    
}